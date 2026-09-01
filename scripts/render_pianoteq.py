#!/usr/bin/env python3
"""Render a directory tree of MIDI files to a mirrored tree of MP3 files."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
from pathlib import Path

from tqdm import tqdm

DEFAULT_PIANOTEQ = Path.home() / ".local/opt/pianoteq/x86-64bit/Pianoteq 8 STAGE"
DEFAULT_PRESET = "NY Steinway D Classical Recording"
DEFAULT_WORKERS = len(os.sched_getaffinity(0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing MIDI files")
    parser.add_argument("output_dir", type=Path, help="Directory to receive MP3 files")
    parser.add_argument(
        "--pianoteq",
        type=Path,
        default=DEFAULT_PIANOTEQ,
        help=f"Pianoteq executable (default: {DEFAULT_PIANOTEQ})",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=f"Pianoteq preset (default: {DEFAULT_PRESET!r})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent Pianoteq processes (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing MP3 files"
    )
    return parser.parse_args()


def render(task: tuple[str, str, str, str]) -> tuple[str, str | None]:
    source_name, output_name, executable, preset = task
    source = Path(source_name)
    output = Path(output_name)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp3")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                executable,
                "--headless",
                "--preset",
                preset,
                "--midi",
                str(source),
                "--mp3",
                str(temporary),
                "--rate",
                "44100",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        temporary.replace(output)
        return source_name, None
    except Exception as error:  # noqa: BLE001 - report failures without stopping the pool
        temporary.unlink(missing_ok=True)
        if isinstance(error, subprocess.CalledProcessError):
            detail = (error.stderr or error.stdout or str(error)).strip()
        else:
            detail = str(error)
        return source_name, detail


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    executable = args.pianoteq.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if input_dir == output_dir:
        raise SystemExit("Input and output directories must differ")
    if not executable.is_file():
        raise SystemExit(f"Pianoteq executable not found: {executable}")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    sources = sorted(
        path for path in input_dir.rglob("*") if path.suffix.lower() == ".mid"
    )
    tasks = []
    skipped = 0
    for source in sources:
        output = (output_dir / source.relative_to(input_dir)).with_suffix(".mp3")
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        tasks.append((str(source), str(output), str(executable), args.preset))

    failures = []
    if tasks:
        with multiprocessing.Pool(processes=args.workers) as pool:
            results = pool.imap_unordered(render, tasks, chunksize=1)
            for source, error in tqdm(results, total=len(tasks), unit="file"):
                if error is not None:
                    failures.append((source, error))

    print(f"Rendered {len(tasks) - len(failures):,}; skipped {skipped:,}")
    if failures:
        for source, error in failures:
            print(f"FAILED {source}: {error}")
        raise SystemExit(f"{len(failures):,} render(s) failed")


if __name__ == "__main__":
    main()
