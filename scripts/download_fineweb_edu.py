#!/usr/bin/env python3
"""Download FineWeb-Edu text into sharded CSVs for the C++ encoder.

The C++ tool (``build/bin/encode_corpus``) only reads CSV and holds all rows
in RAM, so feeding it FineWeb directly is not viable: this script streams the
HF parquet (no full local copy) and cuts it into ``shard_*.csv`` files with a
single ``text`` column. Encode each shard with the C++ tool, then concatenate
the flat int32 ``.bin`` outputs::

    uv run python scripts/download_fineweb_edu.py --max-docs 100000
    for f in data/fineweb-edu/csv/shard_*.csv; do
      b=${f%.csv}.bin
      build/bin/encode_corpus --corpus "$f" --column text \\
        --tokenizer data/tokenizer/tokenizer.json --cache "$b" \\
        --jobs "$(nproc)" --force
    done
    cat data/fineweb-edu/csv/shard_*.bin > data/fineweb-edu/edu_ids.bin

A doc separator (default ``"\\n\\n"``) is appended to every doc at dump time:
neither the Python nor the C++ encoder inserts boundaries between rows, and
without it documents would glue together mid-sentence.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def dump_stream_to_csv_shards(
    stream: Iterable[dict],
    out_dir: Path,
    *,
    column: str = "text",
    docs_per_shard: int = 50_000,
    max_docs: int | None = None,
    separator: str = "\n\n",
    progress_cb: Callable[[int], None] | None = None,
) -> tuple[int, int]:
    """Write ``stream`` rows into ``shard_*.csv`` files.

    Returns:
        ``(num_shards, num_docs)`` written.
    """
    if docs_per_shard < 1:
        raise ValueError("docs_per_shard must be >= 1")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_shards, num_docs = 0, 0
    docs_in_shard = 0
    fh = None
    writer = None

    def close_shard() -> None:
        nonlocal fh, writer
        if fh is not None:
            fh.close()
            fh, writer = None, None

    try:
        for row in stream:
            if max_docs is not None and num_docs >= max_docs:
                break
            text = row.get(column) if isinstance(row, dict) else None
            if not text:
                continue
            if separator and not text.endswith(separator):
                text = text + separator
            if writer is None:
                shard_path = out_dir / f"shard_{num_shards:04d}.csv"
                fh = open(shard_path, "w", newline="", encoding="utf-8")
                writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([column])
                docs_in_shard = 0
            writer.writerow([text])
            num_docs += 1
            docs_in_shard += 1
            if progress_cb is not None:
                progress_cb(num_docs)
            if docs_in_shard >= docs_per_shard:
                close_shard()
                num_shards += 1
                docs_in_shard = 0
        close_shard()
        if docs_in_shard > 0:
            # Last partial shard was closed above but never counted.
            num_shards += 1
        return num_shards, num_docs
    except BaseException:
        close_shard()
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download FineWeb-Edu to CSV shards.")
    p.add_argument("--repo", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--column", default="text")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "fineweb-edu" / "csv",
    )
    p.add_argument("--docs-per-shard", type=int, default=50_000)
    p.add_argument(
        "--max-docs",
        type=int,
        default=100_000,
        help="0 or negative = no limit (whole subset, TBs -- be careful).",
    )
    p.add_argument("--separator", default="\n\n")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from rich.console import Console

    console = Console()
    existing = sorted(args.out_dir.glob("shard_*.csv"))
    if existing and not args.force:
        console.print(
            f"[yellow]{len(existing)} shards already in {args.out_dir}. "
            "Use --force to overwrite or pick another --out-dir.[/yellow]"
        )
        return 1
    if args.force:
        for f in existing:
            f.unlink()

    from datasets import load_dataset

    max_docs = args.max_docs if args.max_docs and args.max_docs > 0 else None
    console.print(
        f"Streaming {args.repo}/{args.subset} [{args.split}] -> {args.out_dir} "
        f"({args.docs_per_shard:,} docs/shard, "
        f"target={[f'{max_docs:,}' if max_docs else 'FULL SUBSET'][0]} docs)"
    )
    ds = load_dataset(args.repo, args.subset, split=args.split, streaming=True)
    t0 = time.time()

    if args.quiet:
        num_shards, num_docs = dump_stream_to_csv_shards(
            ds,
            args.out_dir,
            column=args.column,
            docs_per_shard=args.docs_per_shard,
            max_docs=max_docs,
            separator=args.separator,
        )
    else:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

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
            task = progress.add_task("Downloading docs", total=max_docs)
            num_shards, num_docs = dump_stream_to_csv_shards(
                ds,
                args.out_dir,
                column=args.column,
                docs_per_shard=args.docs_per_shard,
                max_docs=max_docs,
                separator=args.separator,
                progress_cb=lambda done: progress.update(
                    task,
                    completed=done,
                    description=(
                        f"Downloading docs  "
                        f"[dim]{done/max(time.time()-t0,1e-9):,.0f} docs/s[/dim]"
                    ),
                ),
            )

    dt = time.time() - t0
    console.print(
        f"\nDone in {dt/60:.1f} min. docs={num_docs:,} shards={num_shards} "
        f"in {args.out_dir}\n"
        "Next: encode each shard with build/bin/encode_corpus "
        "(--column text), then `cat shard_*.bin > edu_ids.bin`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
