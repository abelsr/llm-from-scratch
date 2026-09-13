import numpy as np
import torch
import torch.nn.functional as F

from llm.gpt import GPT
from llm.data import GPTDataset

ids = np.fromfile("data/tokenizer/corpus_ids.bin", dtype=np.int32)
dataset = GPTDataset(ids, block_size=256)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

model = GPT(
    vocab_size=100_264,
    embed_dim=128,
    num_heads=2,
    num_layers=2,
    max_seq_length=256,
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(10):
    loss = np.inf
    for batch in dataloader:
        x, y = batch # x,y : (batch_size, seq_length) -> (8, 256)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch}: Loss {loss.item()}")