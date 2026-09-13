import csv
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset

from llm.utils.tokenizer import Tokenizer


def load_corpus_rows(csv_path: Path, column: str = "processed") -> list[str]:
    """Read a single text column from a CSV as a list of rows (no joining).

    Uses the standard library ``csv`` module so no pandas dependency is needed.
    """
    rows: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column {column!r} not found in {csv_path}; "
                f"available columns: {reader.fieldnames}"
            )
        for row in reader:
            text = row.get(column)
            if text:
                rows.append(text)
    return rows


def load_corpus(csv_path: Path, column: str = "processed") -> str:
    """Read a single text column and join it into one string."""
    return " ".join(load_corpus_rows(csv_path, column))


def encode_corpus(
    tokenizer: Tokenizer,
    corpus_path: Path,
    cache_path: Path,
    *,
    column: str = "processed",
    force: bool = False,
    progress=None,
    max_rows: int | None = None,
) -> np.ndarray:
    """Encode the corpus into token IDs and cache the result as an int32 array.

    Encoding is done **row by row** (not on the whole ~250MB string at once):
    the BPE merge loop is O(L^2), so a single giant sequence is infeasible, and
    per-row work also lets us report real progress.

    Args:
        tokenizer: any :class:`Tokenizer` exposing ``encode(text) -> list[int]``.
        corpus_path: path to the corpus CSV.
        cache_path: where to store the int32 id array (e.g. ``corpus_ids.bin``).
        column: which CSV column holds the text.
        force: re-encode even if ``cache_path`` already exists.
        progress: optional ``callback(done, total)`` invoked after each row.
        max_rows: if set, only encode the first N rows (useful for a fast,
            smaller training set on CPU).
    """
    if cache_path.exists() and not force:
        print(f"Loading cached encoded corpus from {cache_path}")
        return np.fromfile(cache_path, dtype=np.int32)

    rows = load_corpus_rows(corpus_path, column)
    if max_rows is not None:
        rows = rows[:max_rows]
    total = len(rows)
    all_ids: list[int] = []
    for i, row in enumerate(rows):
        all_ids.extend(tokenizer.encode(row))
        if progress is not None:
            progress(i + 1, total)

    ids_arr = np.array(all_ids, dtype=np.int32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
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
