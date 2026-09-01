#!/usr/bin/env python3
"""Encode an audio directory to a mirrored tree of codec-token files."""

from __future__ import annotations

import argparse
import heapq
import multiprocessing
import os
import queue
from pathlib import Path

import torch
from tqdm import tqdm

from speech_piano_train.codec import encode_file, load_codec

AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing audio files")
    parser.add_argument(
        "output_dir", type=Path, help="Directory to receive token files"
    )
    parser.add_argument("weights", type=Path, help="SpectroStream weights directory")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Codec window batch size (default: 8)"
    )
    parser.add_argument(
        "--core-frames",
        type=int,
        default=250,
        help="Core frames per codec window (default: 250)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing token files"
    )
    return parser.parse_args()


def balanced_shards(
    tasks: list[tuple[Path, Path]], count: int
) -> list[list[tuple[str, str]]]:
    """Greedily balance inputs by compressed file size."""
    shards: list[list[tuple[str, str]]] = [[] for _ in range(count)]
    heap = [(0, index) for index in range(count)]
    sized_tasks = [(source.stat().st_size, source, output) for source, output in tasks]
    for size, source, output in sorted(sized_tasks, reverse=True):
        total, index = heapq.heappop(heap)
        shards[index].append((str(source), str(output)))
        heapq.heappush(heap, (total + size, index))
    return shards


def encode_shard(
    rank: int,
    shard: list[tuple[str, str]],
    weights: str,
    core_frames: int,
    batch_size: int,
    results: multiprocessing.Queue,
) -> None:
    torch.cuda.set_device(rank)
    codec = load_codec(weights, f"cuda:{rank}")
    for source_name, output_name in shard:
        output = Path(output_name)
        temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            encode_file(
                codec,
                source_name,
                temporary,
                core_frames=core_frames,
                batch_size=batch_size,
                resample=True,
            )
            temporary.replace(output)
            results.put((source_name, None))
        except Exception as error:  # noqa: BLE001 - continue encoding other files
            temporary.unlink(missing_ok=True)
            results.put((source_name, str(error)))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if input_dir == output_dir:
        raise SystemExit("Input and output directories must differ")
    for name in ("encoder.safetensors", "decoder.safetensors"):
        if not (weights / name).is_file():
            raise SystemExit(f"Codec weight not found: {weights / name}")
    if args.batch_size < 1 or args.core_frames < 1:
        raise SystemExit("--batch-size and --core-frames must be positive")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise SystemExit("No CUDA devices are visible")

    sources = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    tasks = []
    outputs: dict[Path, Path] = {}
    skipped = 0
    for source in sources:
        relative = source.relative_to(input_dir).with_suffix(".codec.json")
        output = output_dir / relative
        previous = outputs.setdefault(output, source)
        if previous != source:
            raise SystemExit(f"Both {previous} and {source} map to {output}")
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        tasks.append((source, output))

    if not tasks:
        print(f"Encoded 0; skipped {skipped:,}")
        return

    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    shards = balanced_shards(tasks, min(device_count, len(tasks)))
    processes = [
        context.Process(
            target=encode_shard,
            args=(
                rank,
                shard,
                str(weights),
                args.core_frames,
                args.batch_size,
                results,
            ),
        )
        for rank, shard in enumerate(shards)
    ]
    for process in processes:
        process.start()

    failures = []
    completed = 0
    try:
        with tqdm(total=len(tasks), unit="file") as progress:
            while completed < len(tasks):
                try:
                    source, error = results.get(timeout=0.5)
                except queue.Empty:
                    failed_workers = [
                        process
                        for process in processes
                        if process.exitcode not in (None, 0)
                    ]
                    if failed_workers:
                        codes = ", ".join(
                            str(process.exitcode) for process in failed_workers
                        )
                        raise SystemExit(f"Codec worker exited unexpectedly ({codes})")
                    if all(process.exitcode is not None for process in processes):
                        raise SystemExit(
                            "Codec workers exited before reporting all files"
                        )
                    continue
                completed += 1
                progress.update()
                if error is not None:
                    failures.append((source, error))
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join()
        results.close()

    print(f"Encoded {len(tasks) - len(failures):,}; skipped {skipped:,}")
    if failures:
        for source, error in failures:
            print(f"FAILED {source}: {error}")
        raise SystemExit(f"{len(failures):,} encoding(s) failed")


if __name__ == "__main__":
    main()
