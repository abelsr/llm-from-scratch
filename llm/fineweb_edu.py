"""FineWeb-Edu corpus builder compatible with this repo's BPE tokenizer.

Streams ``HuggingFaceFW/fineweb-edu`` (text column ``"text"``), encodes each
document with :class:`llm.utils.bpe_tokenizer.BPETokenizer` and appends the
int32 token IDs to a ``.bin`` cache readable with ``np.memmap`` /
``np.fromfile(..., dtype=np.int32)`` — the same format as
``data/tokenizer/corpus_ids.bin``.

Why streaming + incremental writes instead of :func:`llm.data.encode_corpus`?
That helper accumulates the whole corpus in a Python list first, which is
fine for 45M tokens but blows up RAM for 1-3B tokens. This class flushes to
disk every ``flush_tokens`` tokens, so RAM stays flat.

Typical usage::

    from pathlib import Path
    from llm.fineweb_edu import FineWebEduCorpus
    from llm.utils.bpe_tokenizer import BPETokenizer

    corpus = FineWebEduCorpus(
        tokenizer=BPETokenizer.load("data/tokenizer/tokenizer.json"),
        cache_path=Path("data/fineweb-edu/edu_ids.bin"),
        subset="sample-10BT",
        max_tokens=3_000_000_000,
    )
    ids = corpus.build()          # np.memmap, int32
    dataset = corpus.as_dataset(block_size=256, stride=256)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from llm.data import GPTDataset
from llm.utils.tokenizer import Tokenizer

DEFAULT_REPO = "HuggingFaceFW/fineweb-edu"
DEFAULT_SUBSET = "sample-10BT"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_COLUMN = "text"
DOC_SEPARATOR = "\n\n"


class FineWebEduCorpus:
    """Stream, encode and cache FineWeb-Edu as int32 token IDs.

    Args:
        tokenizer: any :class:`Tokenizer` exposing ``encode(text)``.
        cache_path: where the int32 ``.bin`` lives.
        subset: HF config, e.g. ``"sample-10BT"`` (10B), ``"sample-100BT"``.
        split: HF split (usually ``"train"``).
        text_column: document column (``"text"`` in FineWeb-Edu).
        max_tokens: stop after this many tokens (``None`` = whole stream).
        doc_separator: appended between docs so the model sees boundaries.
        flush_tokens: write to disk every N buffered tokens (RAM control).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        cache_path: str | Path,
        *,
        subset: str = DEFAULT_SUBSET,
        split: str = DEFAULT_SPLIT,
        text_column: str = DEFAULT_TEXT_COLUMN,
        max_tokens: int | None = None,
        doc_separator: str = DOC_SEPARATOR,
        flush_tokens: int = 1_000_000,
    ) -> None:
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive or None")
        self.tokenizer = tokenizer
        self.cache_path = Path(cache_path)
        self.subset = subset
        self.split = split
        self.text_column = text_column
        self.max_tokens = max_tokens
        self.doc_separator = doc_separator
        self.flush_tokens = flush_tokens
        self._sep_ids: list[int] = (
            tokenizer.encode(doc_separator) if doc_separator else []
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def num_cached_tokens(self) -> int:
        """Tokens already in the cache (0 if it does not exist)."""
        if not self.cache_path.exists():
            return 0
        return self.cache_path.stat().st_size // np.dtype(np.int32).itemsize

    def build(
        self,
        *,
        force: bool = False,
        progress: Callable[[int, int | None], None] | None = None,
    ) -> np.memmap:
        """Stream the HF dataset, encode it and return it as a memmap.

        If the cache already exists and ``force`` is False, encoding is
        skipped (same contract as :func:`llm.data.encode_corpus`).
        """
        if self.cache_path.exists() and not force:
            return self.open_memmap()

        from datasets import load_dataset  # lazy: keeps `llm` import light

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any partial file from an interrupted run.
        self.cache_path.write_bytes(b"")

        ds = load_dataset(
            DEFAULT_REPO, self.subset, split=self.split, streaming=True
        )
        buf: list[int] = []
        total = 0
        with open(self.cache_path, "ab") as fh:
            for row in ds:
                text = row.get(self.text_column)
                if not text:
                    continue
                ids = self.tokenizer.encode(text)
                if self._sep_ids:
                    ids = ids + self._sep_ids
                if self.max_tokens is not None:
                    remaining = self.max_tokens - total - len(buf)
                    if remaining <= 0:
                        break
                    ids = ids[:remaining]
                buf.extend(ids)
                if len(buf) >= self.flush_tokens:
                    np.asarray(buf, dtype=np.int32).tofile(fh)
                    total += len(buf)
                    buf.clear()
                    if progress is not None:
                        progress(total, self.max_tokens)
                    if self.max_tokens is not None and total >= self.max_tokens:
                        break
            if buf and (
                self.max_tokens is None or total < self.max_tokens
            ):
                np.asarray(buf, dtype=np.int32).tofile(fh)
                total += len(buf)
                buf.clear()
                if progress is not None:
                    progress(total, self.max_tokens)
        return self.open_memmap()

    def open_memmap(self, mode: str = "r") -> np.memmap:
        """Open the cache without loading it into RAM."""
        return np.memmap(self.cache_path, dtype=np.int32, mode=mode)

    def as_dataset(self, block_size: int = 256, stride: int = 256) -> GPTDataset:
        """Build a :class:`GPTDataset` over the memmap (no RAM copy)."""
        return GPTDataset(self.open_memmap(), block_size=block_size, stride=stride)
