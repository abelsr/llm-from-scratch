#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer from a corpus.

This is a script version of ``notebooks/tokenizer.ipynb``. It:

1. Loads the ``processed`` column from the corpus CSV.
2. Pre-tokenizes, base-splits into bytes and runs the BPE merge loop
   using an inverted index for incremental updates.
3. Builds the final vocabulary and persists it to
   * ``tokenizer.json``  -- self-describing, round-trippable (used by
     :class:`llm.utils.bpe_tokenizer.BPETokenizer`),
   * ``tokenizer_vocabulary.csv`` -- human-readable mapping.

Standard library only (no torch / pandas / numpy required), so it runs
anywhere Python 3.10+ is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------- #
# Repo root on sys.path so `llm` is importable regardless of cwd.
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.utils.bpe_tokenizer import (  # noqa: E402
    BPETokenizer,
    DEFAULT_SPECIAL_TOKENS,
    Token,
    build_tokenizer_from_rules,
    token_to_display,
)

# --------------------------------------------------------------------------- #
# Progress bar (tqdm if available, plain fallback otherwise)
# --------------------------------------------------------------------------- #
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc=None, **_kw):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self._i = 0
            if desc:
                print(f"[{desc}]", flush=True)

        def __iter__(self):
            for x in (self.iterable or []):
                yield x
                self._i += 1
                if self.total and (self._i % max(1, self.total // 20) == 0):
                    self._print()

        def _print(self):
            if self.total:
                pct = 100 * self._i / self.total
                print(f"  {self._i:>7d}/{self.total} ({pct:5.1f}%)", end="\r", flush=True)

        def update(self, n=1):
            self._i += n
            if self.total and (self._i % max(1, self.total // 20) == 0):
                self._print()

        def set_postfix(self, mapping: dict):
            parts = "  ".join(f"{k}={v}" for k, v in mapping.items())
            if self.total:
                pct = 100 * self._i / self.total
                print(f"  {self._i}/{self.total} ({pct:5.1f}%)  {parts}", end="\r", flush=True)

        def close(self):
            if self.total:
                print()


# --------------------------------------------------------------------------- #
# Corpus IO
# --------------------------------------------------------------------------- #
def load_corpus(csv_path: Path, column: str = "processed") -> str:
    """Read a single (text) column from a CSV file and join with spaces.

    Uses the standard library ``csv`` module so no pandas dependency is
    required.
    """
    chunks: list[str] = []
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
                chunks.append(text)
    return " ".join(chunks)


# --------------------------------------------------------------------------- #
# BPE training primitives (mirror the notebook)
# --------------------------------------------------------------------------- #
# Pre-tokenization regex. See `llm/utils/bpe_tokenizer.py` for the full
# rationale — the trailing `\s+` alternative preserves whitespace that the
# original notebook's version silently dropped next to special tokens.
_PRETOKEN_RE = re.compile(r"\s+\S+|\S+|\s+")


def normalize_text(text: str, lowercase: bool = False) -> str:
    """Optional normalization. Byte-level BPE typically keeps the original
    bytes, so by default this is the identity function."""
    if lowercase:
        text = text.lower()
    return text


def pre_tokenize(text: str, special_tokens: Sequence[str]) -> list[str]:
    """Pre-tokenize keeping leading spaces and special tokens intact."""
    special = set(special_tokens)
    if not special:
        return _PRETOKEN_RE.findall(text)
    pattern = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    parts = re.split(pattern, text)
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        if part in special:
            tokens.append(part)
        else:
            tokens.extend(_PRETOKEN_RE.findall(part))
    return tokens


def base_split(
    words_dict: dict[str, int], special_tokens: Sequence[str]
) -> dict[tuple[Token, ...], int]:
    """Split each pre-token into single-byte tokens; special tokens stay."""
    special = set(special_tokens)
    out: dict[tuple[Token, ...], int] = {}
    for token, count in words_dict.items():
        if token in special:
            out[(token,)] = out.get((token,), 0) + count
            continue
        byte_tokens = tuple(bytes([b]) for b in token.encode("utf-8"))
        out[byte_tokens] = out.get(byte_tokens, 0) + count
    return out


def get_pair_stats(
    word_count_dict: dict[tuple[Token, ...], int], special_tokens: set[str]
) -> Counter:
    """Frequency of every adjacent non-special pair."""
    stats: Counter = Counter()
    for tokens, count in word_count_dict.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if any(isinstance(t, str) and t in special_tokens for t in pair):
                continue
            stats[pair] += count
    return stats


def get_inverted_index(
    word_count_dict: dict[tuple[Token, ...], int],
) -> dict[tuple[Token, Token], set]:
    """Map each pair to the set of words (tuples) it appears in."""
    index: dict[tuple[Token, Token], set] = {}
    for tokens in word_count_dict:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            index.setdefault(pair, set()).add(tokens)
    return index


def merge_tokens(tokens: tuple[Token, ...], pair: tuple[Token, Token]) -> tuple[Token, ...]:
    """Apply one (first, second) merge across a single word tuple."""
    first, second = pair
    new_token: Token = first + second
    new_tokens: list[Token] = []
    i, n = 0, len(tokens)
    while i < n:
        if i < n - 1 and tokens[i] == first and tokens[i + 1] == second:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return tuple(new_tokens)


def train_bpe(
    corpus: str,
    special_tokens: Sequence[str],
    *,
    steps: int = 100_000,
    min_count: int = 5,
    max_vocab_size: int = 256 * 1024,
    lowercase: bool = False,
    verbose: bool = True,
) -> tuple[list[tuple[Token, Token]], dict[tuple[Token, ...], int]]:
    """Run the full BPE merge loop.

    Returns:
        (rule_set, final_split_dict)
    """
    special_set = set(special_tokens)
    text = normalize_text(corpus, lowercase=lowercase)

    # Pre-tokenize, count, base split.
    words = pre_tokenize(text, special_tokens)
    words_count = dict(Counter(words))
    split_dict = base_split(words_count, special_tokens)

    stats = get_pair_stats(split_dict, special_set)
    inverted_index = get_inverted_index(split_dict)

    initial_bytes = len(text.encode("utf-8"))
    rule_set: list[tuple[Token, Token]] = []
    cap = max_vocab_size - 256 - len(special_tokens)

    bar = tqdm(total=steps, desc="BPE training", disable=not verbose)
    for step in range(steps):
        if not stats:
            break
        most_common_pair, frequency = stats.most_common(1)[0]

        # Early stopping.
        if frequency < min_count:
            if verbose:
                print(f"Stopping: most common pair frequency {frequency} < min_count {min_count}")
            break
        if len(rule_set) >= cap:
            if verbose:
                print(f"Stopping: reached max_vocab_size={max_vocab_size}")
            break

        first, second = most_common_pair
        new_token: Token = first + second

        # Update only the words that contain this pair (inverted index).
        for old_tokens in list(inverted_index.get(most_common_pair, ())):
            count = split_dict.pop(old_tokens)
            new_tokens = merge_tokens(old_tokens, most_common_pair)
            split_dict[new_tokens] = split_dict.get(new_tokens, 0) + count

            # Remove old pair stats.
            for i in range(len(old_tokens) - 1):
                p = (old_tokens[i], old_tokens[i + 1])
                if any(isinstance(t, str) and t in special_set for t in p):
                    continue
                stats[p] -= count
                if stats[p] <= 0:
                    del stats[p]
                if old_tokens in inverted_index.get(p, ()):
                    inverted_index[p].remove(old_tokens)

            # Add new pair stats.
            for i in range(len(new_tokens) - 1):
                p = (new_tokens[i], new_tokens[i + 1])
                if any(isinstance(t, str) and t in special_set for t in p):
                    continue
                stats[p] += count
                inverted_index.setdefault(p, set()).add(new_tokens)

        rule_set.append(most_common_pair)

        # Progress / metrics.
        if step % 100 == 0 or step == steps - 1:
            current_token_count = sum(len(t) * c for t, c in split_dict.items())
            compression = initial_bytes / current_token_count if current_token_count else 0.0
            bar.set_postfix(
                CR=f"{compression:.2f}x",
                Token=token_to_display(new_token),
                Freq=frequency,
            )
        bar.update(1)
    bar.close()

    return rule_set, split_dict


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_tokenizer(tokenizer: BPETokenizer, output_dir: Path) -> tuple[Path, Path]:
    """Save ``tokenizer`` to ``output_dir`` as JSON (machine) + CSV (human).

    Returns: (json_path, csv_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "tokenizer.json"
    tokenizer.save(json_path)

    csv_path = output_dir / "tokenizer_vocabulary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Token ID", "Token"])
        for tid, display in tokenizer.to_csv_rows():
            writer.writerow([tid, display])

    return json_path, csv_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_corpus = REPO_ROOT / "data" / "notebooks" / "claude_opus_4.6_4.7_reasoning_8.7k.csv"
    default_out = REPO_ROOT / "data" / "tokenizer"

    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer from a corpus CSV."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=default_corpus,
        help=f"Path to the corpus CSV (default: {default_corpus})",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="processed",
        help="CSV column holding the text to train on (default: 'processed')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Directory to write tokenizer artifacts (default: {default_out})",
    )
    parser.add_argument("--steps", type=int, default=100_000, help="Max BPE merge steps")
    parser.add_argument("--min-count", type=int, default=5, help="Stop when most-common pair frequency drops below this")
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=256 * 1024,
        help="Hard cap on total vocabulary size (default: 262144)",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        default=None,
        help="Comma-separated special tokens. Defaults to the notebook's list.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase the corpus before training (byte-level BPE usually keeps original bytes)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the progress bar / extra logging",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.corpus.exists():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    special_tokens = (
        [s for s in args.special_tokens.split(",") if s]
        if args.special_tokens
        else list(DEFAULT_SPECIAL_TOKENS)
    )

    t0 = time.time()
    print(f"Loading corpus from {args.corpus} (column={args.column!r}) ...")
    corpus = load_corpus(args.corpus, column=args.column)
    print(f"  corpus length: {len(corpus):,} chars  ({time.time() - t0:.1f}s)")

    print(f"Training BPE: steps={args.steps}  min_count={args.min_count}  max_vocab={args.max_vocab_size}")
    rule_set, _ = train_bpe(
        corpus,
        special_tokens=special_tokens,
        steps=args.steps,
        min_count=args.min_count,
        max_vocab_size=args.max_vocab_size,
        lowercase=args.lowercase,
        verbose=not args.quiet,
    )
    print(f"  learned {len(rule_set):,} merge rules  ({time.time() - t0:.1f}s)")

    tokenizer = build_tokenizer_from_rules(rule_set, special_tokens)
    json_path, csv_path = save_tokenizer(tokenizer, args.output_dir)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Final vocab size: {tokenizer.vocab_size:,}")

    # Round-trip sanity check.
    sample = "Hello, world!  Is this working?"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"\nRound-trip check:")
    print(f"  in : {sample!r}")
    print(f"  ids: {encoded}")
    print(f"  out: {decoded!r}")
    assert decoded == sample, f"round-trip mismatch: {decoded!r} != {sample!r}"
    print("  OK ✓")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
