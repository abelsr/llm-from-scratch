"""Train GPT with one process per CUDA GPU via torchrun."""

import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
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


def is_main_process() -> bool:
    return dist.get_rank() == 0


def rank_log(message: str) -> None:
    print(f"[rank {dist.get_rank()}] {message}", flush=True)


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
        use_compile = os.environ.get("COMPILE", "0") == "1"
        rank_log("process group initialized")

        if is_main_process():
            console.rule("[bold green]GPT DDP training[/bold green]")
            table = Table(title="Available Devices", header_style="bold magenta")
            table.add_column("Device")
            table.add_column("Name")
            for index in range(torch.cuda.device_count()):
                table.add_row(f"cuda:{index}", torch.cuda.get_device_name(index))
            console.print(table)

        max_tokens = 10_000_000
        rank_log("loading corpus")
        ids = np.fromfile(
            "data/tokenizer/corpus_ids.bin",
            dtype=np.int32,
            count=max_tokens if max_tokens > 0 else -1,
        )
        if len(ids) <= 256:
            raise ValueError("MAX_TOKENS must be greater than the block size (256).")
        dataset = GPTDataset(ids, block_size=256)
        vocab_size = int(ids.max() + 1)
        batch_size_per_gpu = 96
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
        rank_log("dataset and dataloader ready")

        model = GPT(
            vocab_size=vocab_size,
            embed_dim=128,
            num_heads=2,
            num_layers=2,
            max_seq_length=256,
        ).to(device)
        if use_compile:
            rank_log("wrapping model with torch.compile")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        rank_log("wrapping model with DDP")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        rank_log("DDP ready")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        if is_main_process():
            table = Table(title="Training Summary", header_style="bold magenta")
            table.add_column("Component")
            table.add_column("Value")
            table.add_row("Dataset size", f"{len(dataset):,}")
            table.add_row("Corpus tokens", f"{len(ids):,}")
            table.add_row("World size", str(world_size))
            table.add_row("Batch per GPU", str(batch_size_per_gpu))
            table.add_row("Global batch", str(batch_size_per_gpu * world_size))
            table.add_row("Steps per epoch", f"{len(dataloader):,}")
            table.add_row("Workers per rank", str(workers_per_process))
            table.add_row("torch.compile", str(use_compile))
            table.add_row("Model parameters", f"{sum(p.numel() for p in model.parameters()):,}")
            console.print(table)

        epochs = 5
        log_every = 20
        final_loss = float("nan")
        if is_main_process():
            console.print("[bold]Starting DDP training loop...[/bold]")

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

                for x, y in dataloader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(x)
                        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.detach()
                    steps += 1
                    if is_main_process() and steps % log_every == 0:
                        progress.update(
                            epoch_task,
                            advance=log_every,
                            description=(
                                f"Epoch {epoch + 1}/{epochs} "
                                f"loss {loss.detach().item():.4f}"
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
                if is_main_process():
                    progress.console.print(
                        f"  [bold]Epoch {epoch + 1}/{epochs}[/bold] "
                        f"avg loss [yellow]{final_loss:.4f}[/yellow]"
                    )

        if is_main_process():
            console.print(
                Panel(
                    "[bold green]DDP training complete[/bold green]\n"
                    f"Final epoch avg loss: [yellow]{final_loss:.4f}[/yellow] | "
                    f"global batch: {batch_size_per_gpu * world_size:,}",
                    title="Summary",
                    border_style="green",
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
