"""Common tokenizer interface for the from-scratch LLM project.

This module defines an abstract base class shared by every concrete
tokenizer. Concrete tokenizers (e.g. :class:`llm.utils.bpe_tokenizer.BPETokenizer`)
inherit from :class:`Tokenizer` and implement the encoding/decoding and
(anti)serialization logic.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Sequence


class Tokenizer(abc.ABC):
    """Abstract base class for all tokenizers in this project.

    Subclasses must implement :meth:`encode`, :meth:`decode`,
    :meth:`save`, the :attr:`vocab_size` property and provide a
    :meth:`load` classmethod. The default :meth:`tokenize` is a safe
    no-op that subclasses are expected to override.
    """

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Number of distinct tokens in the vocabulary."""

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert ``text`` into a list of token IDs."""

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Convert a sequence of token IDs back into a string."""

    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the tokenizer state to ``path``."""

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        """Load a tokenizer from ``path`` (override in subclasses)."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement a classmethod `load`"
        )

    def tokenize(self, text: str) -> list[str]:
        """Return human-readable tokens for ``text`` (override in subclasses)."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement `tokenize`"
        )

    def __call__(self, text: str) -> list[int]:
        """Convenience alias: ``tokenizer(text)`` is equivalent to ``encode(text)``."""
        return self.encode(text)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size})"
