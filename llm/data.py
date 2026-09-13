from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset

from llm.utils.tokenizer import Tokenizer
from scripts.train_tokenizer import load_corpus

def encode_corpus(
    tokenizer: Tokenizer, corpus_path: Path, cache_path: Path
) -> np.ndarray:
    """
    Encode the corpus using the provided tokenizer and save it to a cache file.

    Args:
        tokenizer: The tokenizer to use for encoding.
        corpus_path: Path to the text corpus file.
        cache_path: Path to save the encoded data.
    """
    if cache_path.exists():
        print(f"Loading cached encoded corpus from {cache_path}")
        return np.fromfile(cache_path, dtype=np.int32)
    text = load_corpus(corpus_path)
    ids = tokenizer.encode(text)
    ids_arr = np.array(ids, dtype=np.int32)
    ids_arr.tofile(cache_path)
    return ids_arr


class GPTDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        block_size: int = 256
    ):
        """
        Initialize the GPTDataset.

        Args:
            data: The encoded data as a numpy array.
            block_size: The size of each block of data to return.
        """
        self.data = data
        self.block_size = block_size
        self.n = len(data) - block_size
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        """
        Get a block of data for the given index.

        Args:
            idx: The index of the block to retrieve.
        Returns:
            The block of data at the specified index.
        """
        x = torch.from_numpy(
            self.data[idx: idx + self.block_size].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[idx + 1: idx + 1 + self.block_size].astype(np.int64)
        )
        return x, y