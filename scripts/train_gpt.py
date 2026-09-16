import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm.gpt import GPT
from llm.data import GPTDataset

console = Console()
console.rule("[bold green]GPT training[/bold green]")

devices = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision("high")

table = Table(title="Available Devices", show_header=True, header_style="bold magenta")
table.add_column("Device", justify="left")
table.add_column("Name", justify="left")
for i in range(devices):
    if i == 0:
        table.add_row(f"cuda:{i} (default)", torch.cuda.get_device_name(i))
    else:
        table.add_row(f"cuda:{i}", torch.cuda.get_device_name(i))
console.print(table)

ids = np.fromfile("data/tokenizer/corpus_ids.bin", dtype=np.int32)
dataset = GPTDataset(ids, block_size=256)
vocab_size = int(ids.max() + 1)
batch_size = 96
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)



table = Table(title="Dataset Summary", show_header=True, header_style="bold magenta")
table.add_column("Component", justify="left")
table.add_column("Value", justify="left")
table.add_row("Dataset Size", f"{len(dataset):,}")
table.add_row("Vocabulary Size", f"{vocab_size:,}")
table.add_row("Batch Size", f"{batch_size:,}")
table.add_row("DataLoader Size", f"{len(dataloader):,}")
table.add_row("Num Workers", f"{dataloader.num_workers:,}")
console.print(table)

model = GPT(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=2,
    num_layers=2,
    max_seq_length=256,
)
model.to(device)
with console.status("[bold green]Compiling model...[/bold green]", spinner="dots"):
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

table = Table(title="Model Summary", show_header=True, header_style="bold magenta")
table.add_column("Component", justify="left")
table.add_column("Value", justify="left")
table.add_row("Vocabulary Size", f"{vocab_size:,}")
table.add_row("Batch Size", f"{batch_size:,}")
table.add_row("DataLoader Size", f"{len(dataloader):,}")
table.add_row("Model Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
table.add_row("Device", f"{device}")
console.print(table)

console.print("[bold]Starting training loop...[/bold]")
total, n = 0.0, 0
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn()
) as progress:
    epochs = 5
    for epoch in range(epochs):
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
        ):
            task = progress.add_task("Training", total=len(dataloader))
            for batch in dataloader:
                x, y = batch  # x,y : (batch_size, seq_length) -> (8, 256)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_value = loss.item()
                total += loss_value
                n += 1
                if n % 10 == 0:
                    progress.update(
                        task,
                        advance=10,
                        description=f"Epoch {epoch + 1}/{epochs} • loss {loss.item():.4f}",
                    )

        progress.console.print(
            f"  [bold]Epoch {epoch + 1}/5[/bold] — avg loss "
            f"[yellow]{total / n:.4f}[/yellow]"
        )

console.print()
console.print(
    Panel(
        "[bold green]Training complete[/bold green]\n"
        f"Final epoch avg loss: [yellow]{total / n:.4f}[/yellow] • "
        f"steps: {len(dataloader):,} "
        f"([dim]{len(dataloader) * batch_size:,} tokens @ {batch_size}×256[/dim])",
        title="Summary",
        border_style="green",
    )
)

console.print("[bold]Testing model generation...[/bold]")

example_text = "GPT are the letters of Generative"
tokens = np.fromfile("data/tokenizer/corpus_ids.bin", dtype=np.int32)
tokens = torch.tensor(tokens, dtype=torch.long, device=device)
model.eval()
with torch.no_grad():
    # Encode the prompt text into token IDs
    prompt_ids = torch.tensor(
        [tokens[i] for i in range(len(tokens)) if tokens[i] in model.tokenizer.vocab],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)  # Add batch dimension

    # Generate new tokens based on the prompt
    generated_ids = model.generate(prompt_ids, max_length=50)

    # Decode the generated token IDs back into text
    generated_text = model.tokenizer.decode(generated_ids[0].cpu().numpy())

console.print(f"Generated text: {generated_text}")
