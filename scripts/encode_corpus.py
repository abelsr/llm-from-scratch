#!/usr/bin/env python3
"""Encode the corpus into token IDs (cached) with a live rich progress bar.

This is a thin CLI wrapper around :func:`llm.data.encode_corpus`. It exists so
you can watch the (long) encoding advance in real time. The heavy lifting is in
``llm/data.py``; this file just adds the progress bar and a couple of stats.

Run it (from the repo root, or inside the docker container) with a TTY so the
bar animates::

    docker exec -it phoson uv run python scripts/encode_corpus.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Repo root on sys.path so `llm` is importable regardless of cwd.
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from llm.data import encode_corpus  # noqa: E402
from llm.utils.bpe_tokenizer import BPETokenizer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_corpus = (
        REPO_ROOT / "data" / "notebooks" / "claude_opus_4.6_4.7_reasoning_8.7k.csv"
    )
    default_tok = REPO_ROOT / "data" / "tokenizer" / "tokenizer.json"
    default_cache = REPO_ROOT / "data" / "tokenizer" / "corpus_ids.bin"

    p = argparse.ArgumentParser(
        description="Encode the corpus into cached token IDs with a progress bar."
    )
    p.add_argument("--corpus", type=Path, default=default_corpus)
    p.add_argument("--column", type=str, default="processed")
    p.add_argument("--tokenizer", type=Path, default=default_tok)
    p.add_argument("--cache", type=Path, default=default_cache)
    p.add_argument("--force", action="store_true", help="Re-encode even if cache exists")
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Only encode the first N rows (smaller, faster training set for CPU)",
    )
    p.add_argument("--quiet", action="store_true", help="No progress bar, plain output")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()

    if not args.corpus.exists():
        console.print(f"[red]corpus not found:[/red] {args.corpus}")
        return 1
    if not args.tokenizer.exists():
        console.print(f"[red]tokenizer not found:[/red] {args.tokenizer}")
        return 1

    console.print(f"Loading tokenizer from {args.tokenizer}")
    tok = BPETokenizer.load(args.tokenizer)
    console.print(f"  vocab_size={tok.vocab_size:,}   merges={len(tok.merges):,}")

    # Fast: just read the CSV to know the total number of rows for the bar.
    from llm.data import load_corpus_rows
    total_rows = len(load_corpus_rows(args.corpus, args.column))
    console.print(f"  corpus rows={total_rows:,}   cache={args.cache}")

    t0 = time.time()

    if args.quiet:
        ids = encode_corpus(
            tok, args.corpus, args.cache, column=args.column,
            force=args.force, max_rows=args.max_rows,
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Encoding corpus", total=total_rows)

            def on_row(done: int, total: int) -> None:
                elapsed = max(time.time() - t0, 1e-9)
                rate = done / elapsed
                progress.update(
                    task,
                    completed=done,
                    total=total,
                    description=f"Encoding corpus  [dim]{rate:.0f} rows/s[/dim]",
                )

            ids = encode_corpus(
                tok,
                args.corpus,
                args.cache,
                column=args.column,
                force=args.force,
                progress=on_row,
                max_rows=args.max_rows,
            )

    dt = time.time() - t0
    n = int(ids.shape[0])
    console.print()
    console.print(
        f"Done in {dt/60:.1f} min.  tokens={n:,}  dtype={ids.dtype}  "
        f"cache={args.cache}"
    )
    console.print(f"-> GPTDataset({n:,} ids, block_size=256) is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
