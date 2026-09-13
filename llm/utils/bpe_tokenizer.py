"""Byte-level BPE tokenizer.

Implements the byte-level Byte-Pair Encoding algorithm that matches the
implementation in ``notebooks/tokenizer.ipynb``:

1. Pre-tokenize on special tokens, keeping leading spaces.
2. Base split every pre-token into single-byte ``bytes`` objects
   (special tokens stay as ``str``).
3. Iteratively merge the most frequent adjacent byte pair until a stop
   criterion is met.
4. Build the final vocabulary: 256 base bytes + special tokens + merges.

Encoding applies the learned merge rules in order; decoding concatenates
the byte representation of every token and UTF-8 decodes it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Sequence

from .tokenizer import Tokenizer


Token = bytes | str
"""A token is either a ``bytes`` object (base byte / merged byte-string)
or a ``str`` (special token kept intact)."""

#: Special tokens the tokenizer never merges across. Mirrors the notebook.
DEFAULT_SPECIAL_TOKENS: list[str] = [
    "<user>",
    "</user>",
    "<assistant>",
    "</assistant>",
    "<system>",
    "</system>",
    "<think>",
    "</think>",
]

#: Default on-disk location of the tokenizer produced by the training script.
DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parents[2] / "data" / "tokenizer" / "tokenizer.json"

# Pre-tokenization regex.
#   * `\s+\S+` attaches a run of leading whitespace to the following word so
#     BPE can learn tokens like " the".
#   * `\S+`    matches a word with no leading whitespace.
#   * `\s+`    matches a run of whitespace on its own. This is the piece the
#     original notebook was missing: right before a special token the
#     whitespace becomes a standalone chunk, and without this alternative it
#     was silently dropped, making the round-trip lossy.
_PRETOKEN_RE = re.compile(r"\s+\S+|\S+|\s+")


# --------------------------------------------------------------------------- #
# Serialization helpers (bytes <-> JSON-safe dict)
# --------------------------------------------------------------------------- #
def _token_to_json(token: Token) -> dict:
    """Encode a token into a JSON-safe dict."""
    if isinstance(token, str):
        return {"type": "str", "text": token}
    return {"type": "bytes", "hex": token.hex()}


def _token_from_json(obj: dict) -> Token:
    """Decode a JSON-safe dict back into a token (bytes or str)."""
    if obj["type"] == "str":
        return obj["text"]
    if obj["type"] == "bytes":
        return bytes.fromhex(obj["hex"])
    raise ValueError(f"Unknown token type: {obj['type']!r}")


def token_to_display(token: Token) -> str:
    """Readable representation of a token for logging / CSV output."""
    if isinstance(token, str):
        return token
    try:
        return token.decode("utf-8")
    except UnicodeDecodeError:
        return f"0x{token.hex()}"


class BPETokenizer(Tokenizer):
    """Byte-level BPE tokenizer.

    Args:
        vocab: Mapping from token (bytes or str) to integer ID.
        special_tokens: Special tokens kept intact (stored as ``str`` keys).
        merges: Optional ordered list of ``(first, second)`` merge rules
            (the "rule_set" produced during training). When supplied,
            :meth:`encode` replays the merges to tokenize text; when
            omitted, :meth:`encode` will raise because it cannot tokenize.

    Note:
        :meth:`decode` only needs ``vocab`` and works whether or not
        ``merges`` are present.
    """

    def __init__(
        self,
        vocab: dict[Token, int],
        special_tokens: list[str] | None = None,
        merges: list[tuple[Token, Token]] | None = None,
    ) -> None:
        self.vocab: dict[Token, int] = dict(vocab)
        self.id_to_token: dict[int, Token] = {v: k for k, v in self.vocab.items()}
        self.special_tokens: list[str] = list(special_tokens or [])
        self.special_set: set[str] = set(self.special_tokens)
        self.merges: list[tuple[Token, Token]] = list(merges) if merges else []
        # Lazy cache: pair -> merge rank (its index in `merges`). Only built on
        # first encode, so loading a tokenizer stays cheap if you never encode.
        self._pair_rank: dict[tuple[Token, Token], int] | None = None

    def _ensure_pair_rank(self) -> dict[tuple[Token, Token], int]:
        """Build (once) the map from merge pair -> its rank (order in `merges`)."""
        if self._pair_rank is None:
            self._pair_rank = {pair: i for i, pair in enumerate(self.merges)}
        return self._pair_rank

    # ------------------------------------------------------------------ #
    # Tokenizer API
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def pre_tokenize(self, text: str) -> list[str]:
        """Split ``text`` into pre-tokens, preserving leading spaces and
        keeping special tokens intact."""
        if not self.special_tokens:
            return _PRETOKEN_RE.findall(text)
        # Build a pattern that keeps special tokens as separate pieces.
        pattern = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
        parts = re.split(pattern, text)
        tokens: list[str] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_set:
                tokens.append(part)
            else:
                tokens.extend(_PRETOKEN_RE.findall(part))
        return tokens

    def _base_split_tokens(self, pre_tokens: list[str]) -> list[Token]:
        """Convert pre-tokens into a list of byte / special-str tokens."""
        out: list[Token] = []
        for t in pre_tokens:
            if t in self.special_set:
                out.append(t)
            else:
                out.extend(bytes([b]) for b in t.encode("utf-8"))
        return out

    @staticmethod
    def _merge_once(tokens: list[Token], pair: tuple[Token, Token]) -> list[Token]:
        """Apply a single (first, second) merge across the whole list."""
        first, second = pair
        new_token: Token = first + second
        out: list[Token] = []
        i, n = 0, len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == first and tokens[i + 1] == second:
                out.append(new_token)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def _merge_to_fixed_point(self, tokens: list[Token]) -> list[Token]:
        """Reproduce the training-time merges efficiently.

        Instead of replaying *all* learned rules (O(#merges * len)), repeatedly
        merge the lowest-rank adjacent pair that is actually present, until no
        further learned pair remains. This is the standard BPE decoding
        procedure and costs O(len^2) instead of O(#merges * len), which matters
        a lot when a tokenizer has ~100k merge rules.
        """
        rank = self._ensure_pair_rank()
        tokens = list(tokens)
        while len(tokens) > 1:
            best_i = -1
            best_rank = None
            for i in range(len(tokens) - 1):
                r = rank.get((tokens[i], tokens[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i < 0:
                break
            first, second = tokens[best_i], tokens[best_i + 1]
            merged = first + second
            # Merge every occurrence of this pair in a single pass.
            out: list[Token] = []
            i, n = 0, len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == first and tokens[i + 1] == second:
                    out.append(merged)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
        return tokens

    def tokenize(self, text: str) -> list[str]:
        """Return the human-readable token sequence for ``text``."""
        pre = self.pre_tokenize(text)
        current = self._merge_to_fixed_point(self._base_split_tokens(pre))
        return [token_to_display(t) for t in current]

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into token IDs using the learned merge order."""
        pre = self.pre_tokenize(text)
        current = self._merge_to_fixed_point(self._base_split_tokens(pre))

        # Map to IDs. Unknown tokens should not happen if `merges` and
        # `vocab` come from the same training run; surface a clear error
        # otherwise.
        ids: list[int] = []
        for tok in current:
            tid = self.vocab.get(tok)
            if tid is None:
                raise KeyError(
                    f"Token {tok!r} is not in the vocabulary. "
                    "Are you sure this tokenizer was trained on text "
                    "with the same normalization/special tokens?"
                )
            ids.append(tid)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token IDs back into a UTF-8 string."""
        parts: list[bytes] = []
        for idx in ids:
            tok = self.id_to_token.get(idx)
            if tok is None:
                raise KeyError(f"Token ID {idx} is not in the vocabulary")
            if isinstance(tok, bytes):
                parts.append(tok)
            else:
                parts.append(tok.encode("utf-8"))
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """Save to a JSON file (self-describing, round-trippable)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "special_tokens": self.special_tokens,
            "merges": [[_token_to_json(a), _token_to_json(b)] for a, b in self.merges],
            "id_to_token": [_token_to_json(t) for _, t in sorted(self.id_to_token.items())],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer produced by :meth:`save` (or the training script)."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version", 1) != 1:
            raise ValueError(f"Unsupported tokenizer version: {payload.get('version')}")
        special_tokens = payload.get("special_tokens", [])
        merges = [
            (_token_from_json(a), _token_from_json(b)) for a, b in payload.get("merges", [])
        ]
        id_to_token: dict[int, Token] = {
            i: _token_from_json(tok) for i, tok in enumerate(payload["id_to_token"])
        }
        vocab = {tok: i for i, tok in id_to_token.items()}
        return cls(vocab=vocab, special_tokens=special_tokens, merges=merges)

    def to_csv_rows(self) -> list[tuple[int, str]]:
        """Return ``(id, display)`` rows ordered by id (for human-readable CSV)."""
        return [(i, token_to_display(self.id_to_token[i])) for i in sorted(self.id_to_token)]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_vocab_and_rules(
        cls,
        vocab: dict[Token, int],
        special_tokens: list[str],
        merges: list[tuple[Token, Token]],
    ) -> "BPETokenizer":
        """Explicit constructor mirroring the notebook's `build_tokenizer`."""
        return cls(vocab=vocab, special_tokens=special_tokens, merges=merges)


def build_tokenizer_from_rules(
    rule_set: list[tuple[Token, Token]],
    special_tokens: list[str],
) -> BPETokenizer:
    """Build a :class:`BPETokenizer` from a raw merge ``rule_set``.

    Mirrors ``build_tokenizer`` in the notebook:
    base bytes (0-255) + special tokens + merged tokens.
    """
    vocab: dict[Token, int] = {}
    for i in range(256):
        vocab[bytes([i])] = i
    curr_id = 256
    for st in special_tokens:
        vocab[st] = curr_id
        curr_id += 1
    for a, b in rule_set:
        new_token: Token = a + b
        if new_token not in vocab:
            vocab[new_token] = curr_id
            curr_id += 1
    return BPETokenizer(vocab=vocab, special_tokens=special_tokens, merges=rule_set)


__all__ = [
    "Token",
    "DEFAULT_SPECIAL_TOKENS",
    "DEFAULT_TOKENIZER_PATH",
    "BPETokenizer",
    "build_tokenizer_from_rules",
    "token_to_display",
]
