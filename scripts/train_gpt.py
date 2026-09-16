import os
import sys

from rich.console import Console
from rich.panel import Panel
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
console = Console()

console.rule("[bold green]GPT training[/bold green]")
ids = np.fromfile("data/tokenizer/corpus_ids.bin", dtype=np.int32)
dataset = GPTDataset(ids, block_size=256)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
console.print(
    f"[cyan]Dataset:[/cyan] {len(dataset):,} samples"
)

model = GPT(
    vocab_size=100_264,
    embed_dim=128,
    num_heads=2,
    num_layers=2,
    max_seq_length=256,
)
model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
console.print(
    f"[cyan]Model:[/cyan] GPT(vocab=100_264, dim=128, heads=2, layers=2, ctx=256) • "
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
    task = progress.add_task("Training", total=10 * len(dataloader))
    for epoch in range(5):
        total, n = 0.0, 0
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
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
        f"steps: {10 * len(dataloader):,} "
        f"([dim]{10 * len(dataloader) * 8 * 256:,} tokens @ 8×256[/dim])",
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
