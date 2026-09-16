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

import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llm.gpt import GPT
from llm.data import GPTDataset

console = Console()

devices = torch.cuda.device_count()
# Print rich table with device info
table = Table(title="Available Devices", show_header=True, header_style="bold magenta")
table.add_column("Device", justify="left")
table.add_column("Name", justify="left")
for i in range(devices):
    table.add_row(f"cuda:{i}", torch.cuda.get_device_name(i))
console.print(table)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

console.rule("[bold green]GPT training[/bold green]")
ids = np.fromfile("data/tokenizer/corpus_ids.bin", dtype=np.int32)
dataset = GPTDataset(ids, block_size=256)
vocab_size = int(ids.max() + 1)
batch_size = 96
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
console.print(
    f"[cyan]Dataset:[/cyan] {len(dataset):,} samples"
)

console.print(f"[cyan]Vocabulary size:[/cyan] {vocab_size:,} tokens")
console.print(f"[cyan]Batch size:[/cyan] {batch_size:,}")
console.print(f"[cyan]DataLoader size:[/cyan] {len(dataloader):,}")

model = GPT(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=2,
    num_layers=2,
    max_seq_length=256,
)
model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
console.print(
    f"[cyan]Model:[/cyan] GPT(vocab={vocab_size}, dim=128, heads=2, layers=2, ctx=256) • "
    f"[bold]{sum(p.numel() for p in model.parameters()):,}[/bold] params"
)
console.print(
    f"[cyan]Device:[/cyan] {device} • "
    f"[dim]{torch.cuda.get_device_name(device) if device.type == 'cuda' else ''}[/dim]"
)
console.print("[cyan]Optimizer:[/cyan] AdamW(lr=1e-3) • grad_clip=1.0 • epochs=10")

console.print("[bold]Starting training loop...[/bold]")
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn()
) as progress:
    task = progress.add_task("Training", total=len(dataloader))
    for epoch in range(5):
        total, n = 0.0, 0
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
        ):
            for batch in dataloader:
                x, y = batch  # x,y : (batch_size, seq_length) -> (8, 256)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item()
                n += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"Epoch {epoch + 1}/10 • loss {loss.item():.4f}",
                )

        progress.console.print(
            f"  [bold]Epoch {epoch + 1}/10[/bold] — avg loss "
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
