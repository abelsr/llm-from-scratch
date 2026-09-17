#!/usr/bin/env python3
"""Download + encode FineWeb-Edu with this repo's BPE tokenizer.

Streams text from HuggingFace (no full download), encodes doc by doc and
appends int32 IDs to a `.bin` cache with flat RAM usage.

Examples::

    # 100M-token smoke test (~minutes, validates the pipeline)
    uv run python scripts/prepare_fineweb_edu.py --max-tokens 100000000

    # 3B Chinchilla-scale for your 147M model (~hours, BPE python is slow)
    uv run python scripts/prepare_fineweb_edu.py --max-tokens 3000000000

    # Full sample-10BT subset
    uv run python scripts/prepare_fineweb_edu.py --subset sample-10BT --max-tokens 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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

from llm.fineweb_edu import (  # noqa: E402
    DEFAULT_SPLIT,
    DEFAULT_SUBSET,
    FineWebEduCorpus,
)
from llm.utils.bpe_tokenizer import BPETokenizer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a FineWeb-Edu .bin cache.")
    p.add_argument("--subset", default=DEFAULT_SUBSET)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--column", default="text")
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=REPO_ROOT / "data" / "tokenizer" / "tokenizer.json",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "data" / "fineweb-edu" / "edu_ids.bin",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=100_000_000,
        help="0 or negative = whole stream (can be TBs, be careful).",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    if not args.tokenizer.exists():
        console.print(f"[red]tokenizer not found:[/red] {args.tokenizer}")
        return 1

    tok = BPETokenizer.load(args.tokenizer)
    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    corpus = FineWebEduCorpus(
        tokenizer=tok,
        cache_path=args.cache,
        subset=args.subset,
        split=args.split,
        text_column=args.column,
        max_tokens=max_tokens,
    )
    console.print(
        f"tokenizer vocab={tok.vocab_size:,}  subset={args.subset}  "
        f"target={[f'{max_tokens:,}' if max_tokens else 'FULL STREAM'][0]} tokens"
    )
    t0 = time.time()

    def on_progress(done: int, total: int | None) -> None:
        pass  # wired to the rich bar below

    if args.quiet:
        ids = corpus.build(force=args.force)
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
        ) as progress:
            task = progress.add_task("Encoding FineWeb-Edu", total=max_tokens)
            ids = corpus.build(
                force=args.force,
                progress=lambda done, _: progress.update(
                    task,
                    completed=done,
                    description=f"Encoding FineWeb-Edu  [dim]{done/max(time.time()-t0,1e-9):,.0f} tok/s[/dim]",
                ),
            )
            on_progress(0, None)

    dt = time.time() - t0
    console.print(
        f"\nDone in {dt/60:.1f} min. tokens={len(ids):,} cache={args.cache}\n"
        f"-> use it with GPTDataset(ids, block_size=256, stride=256), "
        f"or np.memmap('{args.cache}', dtype='int32') for zero-RAM reads."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
