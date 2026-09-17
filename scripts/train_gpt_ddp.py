"""Train GPT with one process per CUDA GPU via torchrun."""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm.data import GPTDataset
from llm.gpt import GPT
from llm.utils.bpe_tokenizer import BPETokenizer


def run_validation(
    raw_model: torch.nn.Module,
    val_loader: DataLoader | None,
    device: torch.device,
) -> float:
    """Mean cross-entropy over the held-out split. Returns NaN if no val set."""
    if val_loader is None or len(val_loader) == 0:
        return float("nan")
    raw_model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = raw_model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            total += loss.item()
            count += 1
    raw_model.train()
    return total / max(count, 1)


def run_generation_demo(
    raw_model: torch.nn.Module,
    device: torch.device,
    console: Console,
    max_new_tokens: int = 100,
) -> None:
    """Generate a few prompts with the trained model (rank 0 only)."""
    try:
        tokenizer = BPETokenizer.load("data/tokenizer/tokenizer.json")
    except Exception as exc:  # noqa: BLE001 - demo must never crash training
        console.print(f"[yellow]Skip generation: no pude cargar tokenizer ({exc})[/yellow]")
        return
    prompts = [
        "<user>\nHello, who are you?\n</assistant>\n",
        "<user>\nExplain gravity in one sentence.\n</assistant>\n",
    ]
    raw_model.eval()
    console.print(Panel("[bold green]Generation demo[/bold green]", border_style="green"))
    for prompt in prompts:
        try:
            prompt_ids = torch.tensor(
                [tokenizer.encode(prompt)], dtype=torch.long, device=device
            )
            out = raw_model.generate(
                prompt_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50
            )
            text = tokenizer.decode(out[0].cpu().tolist())
            console.print(f"[bold cyan]PROMPT:[/bold cyan] {prompt!r}")
            console.print(f"[green]GEN:[/green] {text[-800:]}\n")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Prompt falló: {exc}[/yellow]")


def is_main_process() -> bool:
    return dist.get_rank() == 0


def rank_log(message: str) -> None:
    print(f"[rank {dist.get_rank()}] {message}", flush=True)


def unwrap_model(model: DDP) -> torch.nn.Module:
    """Return the original GPT module for portable checkpoints."""
    return getattr(model.module, "_orig_mod", model.module)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA GPUs.")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Launch with torchrun, for example: "
            "torchrun --standalone --nproc_per_node=2 scripts/train_gpt_ddp.py"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    console = Console()

    try:
        torch.set_float32_matmul_precision("high")
        use_compile = os.environ.get("COMPILE", "1") == "1"
        seed = int(os.environ.get("SEED", "1337"))
        base_lr = float(os.environ.get("LR", "3e-4"))
        warmup_steps = int(os.environ.get("WARMUP_STEPS", "500"))
        block_size = int(os.environ.get("BLOCK_SIZE", "256"))
        stride = int(os.environ.get("STRIDE", "256"))
        batch_size_per_gpu = int(os.environ.get("BATCH_PER_GPU", "80"))
        val_fraction = float(os.environ.get("VAL_FRACTION", "0.01"))
        gen_tokens = int(os.environ.get("GEN_TOKENS", "100"))
        torch.manual_seed(seed + dist.get_rank())
        # rank_log("process group initialized")

        if is_main_process():
            console.rule("[bold green]GPT DDP training[/bold green]")
            table = Table(title="Available Devices", header_style="bold magenta")
            table.add_column("Device")
            table.add_column("Name")
            for index in range(torch.cuda.device_count()):
                table.add_row(f"cuda:{index}", torch.cuda.get_device_name(index))
            console.print(table)
            EXP_NAME = Prompt.ask(
                "[bold]Enter experiment name[/bold]", default="gpt_ddp_experiment"
            )
            if not EXP_NAME:
                raise ValueError("Experiment name cannot be empty. Please provide a valid name.")

        # max_tokens = int(os.environ.get("MAX_TOKENS", "500000"))
        max_tokens = int(os.environ.get("MAX_TOKENS", "-1"))  # -1: use all tokens
        # rank_log("loading corpus")
        ids = np.fromfile(
            "data/tokenizer/corpus_ids.bin",
            dtype=np.int32,
            count=max_tokens if max_tokens > 0 else -1,
        )
        if len(ids) <= block_size:
            raise ValueError("MAX_TOKENS must be greater than the block size (256).")
        # Split train/val por posición: el val es el 1% final (sin shuffle).
        # Así la val loss mide generalización, no memorización.
        n_val_tokens = int(len(ids) * val_fraction)
        if n_val_tokens > block_size + 1:
            train_ids = ids[:-n_val_tokens]
            val_ids = ids[-n_val_tokens:]
        else:
            train_ids, val_ids = ids, None
        dataset = GPTDataset(train_ids, block_size=block_size, stride=stride)
        val_dataset = (
            GPTDataset(val_ids, block_size=block_size, stride=stride)
            if val_ids is not None
            else None
        )
        vocab_size = int(ids.max() + 1)
        expected_init_loss = math.log(vocab_size)
        workers_per_process = max(1, min(8, (os.cpu_count() or 1) // world_size))
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            sampler=sampler,
            num_workers=workers_per_process,
            pin_memory=True,
            persistent_workers=True,
        )
        # Val solo en rank 0 para no complicar la sincronización DDP.
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size_per_gpu,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            if val_dataset is not None and is_main_process()
            else None
        )
        # rank_log("dataset and dataloader ready")

        model = GPT(
            vocab_size=vocab_size,
            embed_dim=1024,
            num_heads=8,
            num_kv_heads=2,
            num_layers=4,
            max_seq_length=256,
        ).to(device)
        if use_compile:
            # rank_log("wrapping model with torch.compile")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        # rank_log("wrapping model with DDP")
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank, 
            static_graph=True,
            gradient_as_bucket_view=True,
        )
        # rank_log("DDP ready")
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, fused=True)

        tokens_per_step = batch_size_per_gpu * block_size * world_size
        # len(dataloader) es steps por rank (= optimizer steps globales en DDP).
        total_tokens_per_epoch = len(dataloader) * tokens_per_step

        if is_main_process():
            table = Table(title="Training Summary", header_style="bold magenta")
            table.add_column("Component")
            table.add_column("Value")
            table.add_row("Dataset size", f"{len(dataset):,}")
            table.add_row("Corpus tokens", f"{len(ids):,}")
            table.add_row(
                "Val tokens",
                f"{len(val_ids):,}" if val_ids is not None else "none",
            )
            table.add_row("Vocab size", f"{vocab_size:,}")
            table.add_row("Expected init loss ln(V)", f"{expected_init_loss:.4f}")
            table.add_row("Block size / stride", f"{block_size} / {stride}")
            table.add_row("World size", str(world_size))
            table.add_row("Batch per GPU", str(batch_size_per_gpu))
            table.add_row("Global batch", str(batch_size_per_gpu * world_size))
            table.add_row("Steps per epoch", f"{len(dataloader):,}")
            table.add_row("Tokens/step (global)", f"{tokens_per_step:,}")
            table.add_row("Tokens/epoch (global)", f"{total_tokens_per_epoch:,}")
            table.add_row("Base LR / warmup", f"{base_lr:g} / {warmup_steps:,}")
            table.add_row("Workers per rank", str(workers_per_process))
            table.add_row("torch.compile", str(use_compile))
            table.add_row("Model parameters", f"{sum(p.numel() for p in model.parameters()):,}")
            console.print(table)

        epochs = int(os.environ.get("EPOCHS", "5"))
        log_every = int(os.environ.get("LOG_EVERY", "20"))
        final_loss = float("nan")
        global_step = 0
        steady_tokens_per_sec = float("nan")
        if is_main_process():
            console.print("[bold]Starting DDP training loop...[/bold]")
            console.print(
                f"[dim]Warmup: {warmup_steps:,} steps hasta LR={base_lr:g} • "
                f"throughput estable se mide tras el warmup[/dim]"
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=not is_main_process(),
        ) as progress:
            epoch_task = progress.add_task("Epoch", total=len(dataloader))
            global_task = progress.add_task(
                "Global training", total=epochs * len(dataloader)
            )
            for epoch in range(epochs):
                if is_main_process():
                    progress.reset(
                        epoch_task,
                        total=len(dataloader),
                        description=f"Epoch {epoch + 1}/{epochs}",
                    )
                sampler.set_epoch(epoch)
                model.train()
                total_loss = torch.zeros((), device=device)
                steps = 0
                interval_start = time.perf_counter()
                interval_tokens = 0

                for x, y in dataloader:
                    # Linear warmup: evita picos de loss al inicio con LR alta.
                    global_step += 1
                    if warmup_steps > 0 and global_step <= warmup_steps:
                        warmup_lr = base_lr * global_step / warmup_steps
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = warmup_lr
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(x)
                        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                    loss_value = loss.detach()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss_value
                    steps += 1
                    interval_tokens += tokens_per_step
                    if is_main_process() and steps % log_every == 0:
                        now = time.perf_counter()
                        interval_sec = max(now - interval_start, 1e-6)
                        # Solo considera estable tras el warmup.
                        if global_step > warmup_steps:
                            steady_tokens_per_sec = interval_tokens / interval_sec
                        tok_s = interval_tokens / interval_sec
                        interval_start = now
                        interval_tokens = 0
                        progress.update(
                            epoch_task,
                            advance=log_every,
                            description=(
                                f"Epoch {epoch + 1}/{epochs} "
                                f"loss {loss_value.item():.4f} "
                                f"{tok_s:,.0f} tok/s"
                            ),
                        )
                        progress.update(global_task, advance=log_every)

                if is_main_process() and steps % log_every:
                    remaining_steps = steps % log_every
                    progress.update(epoch_task, advance=remaining_steps)
                    progress.update(global_task, advance=remaining_steps)

                metrics = torch.stack((total_loss, torch.tensor(steps, device=device)))
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                final_loss = (metrics[0] / metrics[1]).item()
                val_loss = float("nan")
                if is_main_process() and val_loader is not None:
                    val_loss = run_validation(unwrap_model(model), val_loader, device)
                if is_main_process():
                    msg = (
                        f"  [bold]Epoch {epoch + 1}/{epochs}[/bold] "
                        f"train [yellow]{final_loss:.4f}[/yellow]"
                    )
                    if not math.isnan(val_loss):
                        msg += f" • val [magenta]{val_loss:.4f}[/magenta]"
                        if val_loss > final_loss + 1.0:
                            msg += " [dim](gap train/val alto: posible overfit)[/dim]"
                    if not math.isnan(steady_tokens_per_sec):
                        msg += f" • steady [cyan]{steady_tokens_per_sec:,.0f} tok/s[/cyan]"
                    # Alerta temprana: si la loss supera 2x la inicial esperada,
                    # probablemente diverge (LR alta o bug de logits).
                    if final_loss > 2 * expected_init_loss:
                        msg += (
                            f" [bold red]⚠ loss muy por encima de "
                            f"ln(V)={expected_init_loss:.2f}[/bold red]"
                        )
                    progress.console.print(msg)
                
                if is_main_process():
                    checkpoint_path = f"checkpoints/{EXP_NAME}/gpt_epoch_{epoch + 1}.pt"
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": final_loss,
                            "val_loss": val_loss,
                            "config": {
                                "vocab_size": vocab_size,
                                "embed_dim": 1024,
                                "num_heads": 8,
                                "num_kv_heads": 2,
                                "num_layers": 4,
                                "max_seq_length": block_size,
                                "block_size": block_size,
                                "stride": stride,
                                "batch_size_per_gpu": batch_size_per_gpu,
                                "world_size": world_size,
                                "max_tokens": max_tokens,
                                "base_lr": base_lr,
                                "warmup_steps": warmup_steps,
                            },
                        },
                        checkpoint_path,
                    )
                    progress.console.print(
                        f"  [bold]Checkpoint saved:[/bold] {checkpoint_path}"
                    )

        if is_main_process():
            console.print(
                Panel(
                    f"[bold green]Training complete[/bold green]\n"
                    f"train loss: [yellow]{final_loss:.4f}[/yellow]\n"
                    + (
                        f"val loss: [magenta]{val_loss:.4f}[/magenta]\n"
                        if not math.isnan(val_loss)
                        else ""
                    )
                    + (
                        f"steady throughput: [cyan]{steady_tokens_per_sec:,.0f} tok/s[/cyan]"
                        if not math.isnan(steady_tokens_per_sec)
                        else ""
                    ),
                    title="Summary",
                    border_style="green",
                )
            )
            run_generation_demo(
                unwrap_model(model), device, console, max_new_tokens=gen_tokens
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
