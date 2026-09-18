from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from speech_piano_train.config import DataConfig, ModelConfig
from speech_piano_train.mel import CONFIG as MEL_CONFIG
from speech_piano_train.midi import parse_piano_line
from speech_piano_train.pianoteq import PianoteqRenderer
from speech_piano_train.prepared_stream import (
    INDEX_DTYPE,
    MIDI_KIND,
    SEGMENT_DTYPE,
    TOKEN_DTYPE,
    MidiRun,
    PreparedStreamWriter,
    TokenRun,
)


@dataclass(frozen=True)
class Document:
    youtube_id: str
    modality: str
    path: Path


_WORKER_TOKENIZER: Any = None


@dataclass(frozen=True)
class CompiledDocument:
    document: Document
    runs: tuple[TokenRun | MidiRun, ...]


def prepare_data(
    data_config: DataConfig,
    model_config: ModelConfig,
    *,
    tokenizer: Any = None,
) -> dict[str, Any]:
    dataset_dir = data_config.dataset_path
    output_dir = data_config.prepared_path
    documents_dir = dataset_dir / "documents"
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        # Releases before 0.5 stored the manifest beside the documents.
        manifest_path = documents_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = ordered_documents(
        documents_dir,
        manifest,
        variant=data_config.variant,
        seed=data_config.seed,
    )

    output_dir.mkdir(parents=True)

    tokenizer = tokenizer or load_tokenizer(model_config)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")

    counts = {"interleaved": 0, "speech": 0, "midi": 0}
    with (
        PreparedStreamWriter(output_dir, data_config.sequence_length) as writer,
        (output_dir / "order.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as order_file,
    ):
        iterator = compile_documents(
            documents,
            model_config,
            representation=data_config.representation,
            seed=data_config.seed,
            workers=data_config.workers,
            tokenizer=tokenizer,
        )
        description = (
            "Tokenizing de-interleaved speech and MIDI documents"
            if data_config.variant == "separated"
            else "Tokenizing interleaved documents"
        )
        for compiled in tqdm(
            iterator,
            total=len(documents),
            desc=description,
            unit="documents",
        ):
            document = compiled.document
            logical_start = writer.logical_positions
            token_start = writer.tokens
            for run in compiled.runs:
                writer.append(run)
            writer.append(TokenRun([eos_token_id]))
            counts[document.modality] += 1
            order_file.write(
                json.dumps(
                    {
                        "youtube_id": document.youtube_id,
                        "modality": document.modality,
                        "path": document.path.relative_to(documents_dir).as_posix(),
                        "logical_start": logical_start,
                        "logical_count": writer.logical_positions - logical_start,
                        "token_start": token_start,
                        "token_count": writer.tokens - token_start,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        stream_counts = writer.close()

    metadata = {
        "schema_version": 1,
        "variant": data_config.variant,
        "representation": data_config.representation,
        "seed": data_config.seed,
        "sequence_length": data_config.sequence_length,
        "tokenizer": model_config.name,
        "eos_token_id": eos_token_id,
        "documents_by_modality": counts,
        **stream_counts,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def ordered_documents(
    documents_dir: Path,
    manifest: dict[str, Any],
    *,
    variant: str,
    seed: int,
) -> list[Document]:
    items = [item for item in manifest["items"] if item["split"] == "train"]
    benchmark_ids = set(
        manifest.get("splits", {}).get("benchmark", {}).get("source_youtube_ids", [])
    )
    if any(
        item.get("is_benchmark_source") or item["youtube_id"] in benchmark_ids
        for item in items
    ):
        raise ValueError("Training split contains a benchmark-source item")

    documents: list[Document] = []
    if variant == "interleaved":
        documents = [
            Document(
                youtube_id=item["youtube_id"],
                modality="interleaved",
                path=document_path(
                    documents_dir,
                    item,
                    field="document",
                    kind="interleaved",
                ),
            )
            for item in items
        ]
    elif variant == "separated":
        for item in items:
            for modality, field, kind in (
                (
                    "speech",
                    "deinterleaved_speech_document",
                    "deinterleaved-speech",
                ),
                (
                    "midi",
                    "deinterleaved_midi_document",
                    "deinterleaved-midi",
                ),
            ):
                path = document_path(
                    documents_dir,
                    item,
                    field=field,
                    kind=kind,
                    legacy_field=f"{modality}_document",
                )
                if path.stat().st_size:
                    documents.append(
                        Document(
                            youtube_id=item["youtube_id"],
                            modality=modality,
                            path=path,
                        )
                    )
    else:
        raise ValueError(f"Unknown data variant: {variant}")

    rng = random.Random(seed)
    for _ in range(100):
        rng.shuffle(documents)
        if variant != "separated" or not has_adjacent_pair(documents):
            break
    else:
        raise RuntimeError("Could not separate paired speech and MIDI documents")
    return documents


def document_path(
    documents_dir: Path,
    item: dict[str, Any],
    *,
    field: str,
    kind: str,
    legacy_field: str | None = None,
) -> Path:
    """Resolve explicit paths from old manifests or derived v0.5 paths."""
    for candidate in (field, legacy_field):
        if candidate is not None and candidate in item:
            return documents_dir / item[candidate]

    youtube_id = item["youtube_id"]
    return documents_dir / youtube_id[:2] / f"{youtube_id}.{kind}.doc.txt"


def has_adjacent_pair(documents: list[Document]) -> bool:
    return any(
        first.youtube_id == second.youtube_id for first, second in pairwise(documents)
    )


def load_tokenizer(model_config: ModelConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_config.name)
    tokenizer.model_max_length = 2**60
    return tokenizer


def compile_documents(
    documents: list[Document],
    model_config: ModelConfig,
    *,
    representation: str,
    seed: int,
    workers: int,
    tokenizer: Any,
) -> Iterator[CompiledDocument]:
    if workers == 1:
        for document in documents:
            yield compile_document(
                document,
                tokenizer,
                representation=representation,
                seed=seed,
            )
        return

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(model_config.name,),
    ) as executor:
        arguments = ((document, representation, seed) for document in documents)
        yield from executor.map(_compile_worker, arguments, chunksize=1)


def compile_document(
    document: Document,
    tokenizer: Any,
    *,
    representation: str,
    seed: int,
) -> CompiledDocument:
    text = document.path.read_text(encoding="utf-8")
    if representation == "midi_text":
        runs = (TokenRun(tokenizer.encode(text, add_special_tokens=False)),)
    elif representation == "mel":
        runs = tuple(
            compile_mel_runs(
                text,
                tokenizer,
                seed=seed,
                document_id=document.youtube_id,
            )
        )
    else:
        raise ValueError(f"Unknown representation: {representation}")
    return CompiledDocument(document, runs)


def compile_mel_runs(
    text: str,
    tokenizer: Any,
    *,
    seed: int,
    document_id: str,
) -> Iterator[TokenRun | MidiRun]:
    text_buffer = ""
    piano_index = 0
    for line_with_ending in text.splitlines(keepends=True):
        line = line_with_ending.rstrip("\r\n")
        line_ending = line_with_ending[len(line) :]
        if not line.startswith("<piano>"):
            text_buffer += line_with_ending
            continue

        block = parse_piano_line(line)
        text_buffer += "<piano>"
        yield TokenRun(tokenizer.encode(text_buffer, add_special_tokens=False))
        text_buffer = f"</piano>{line_ending}"

        tail_ms = random_tail_ms(seed, document_id, piano_index)
        target_samples = (block.end_ms + tail_ms) * MEL_CONFIG.sample_rate // 1_000
        mel_frames = target_samples // MEL_CONFIG.hop_length
        logical_length = math.ceil(mel_frames / MEL_CONFIG.frames_per_position)
        yield MidiRun(line, target_samples, logical_length)
        piano_index += 1

    if text_buffer:
        yield TokenRun(tokenizer.encode(text_buffer, add_special_tokens=False))


def random_tail_ms(seed: int, document_id: str, piano_index: int) -> int:
    payload = f"{seed}\0{document_id}\0{piano_index}".encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value % 101 * 10


def _init_worker(name: str) -> None:
    global _WORKER_TOKENIZER
    from transformers import AutoTokenizer

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(name)
    _WORKER_TOKENIZER.model_max_length = 2**60


def _compile_worker(arguments: tuple[Document, str, int]) -> CompiledDocument:
    document, representation, seed = arguments
    return compile_document(
        document,
        _WORKER_TOKENIZER,
        representation=representation,
        seed=seed,
    )


class TokenDataset(torch.utils.data.Dataset[dict[str, torch.Tensor | None]]):
    def __init__(
        self,
        root: str | Path,
        *,
        pianoteq: str | None = None,
        renderer: Any = None,
    ) -> None:
        self.root = Path(root)
        self.metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        self.sequence_length = int(self.metadata["sequence_length"])
        self.representation = self.metadata["representation"]
        self.tokens = mmap_or_empty(
            self.root / "tokens.bin", TOKEN_DTYPE, int(self.metadata["tokens"])
        )
        self.segments = mmap_or_empty(
            self.root / "segments.bin",
            SEGMENT_DTYPE,
            int(self.metadata["segments"]),
        )
        self.index = mmap_or_empty(
            self.root / "index.bin", INDEX_DTYPE, int(self.metadata["sequences"])
        )
        self.piano_blocks = None
        if self.metadata["piano_block_bytes"]:
            self.piano_blocks = np.memmap(
                self.root / "piano_blocks.bin",
                dtype=np.uint8,
                mode="r",
                shape=(int(self.metadata["piano_block_bytes"]),),
            )
        self.pianoteq = pianoteq
        self._renderer = renderer

    def __len__(self) -> int:
        return int(self.metadata["sequences"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | None]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cursor = self.index[index]
        segment_id = int(cursor["segment_id"])
        segment_offset = int(cursor["offset"])
        input_ids = torch.empty(self.sequence_length, dtype=torch.long)
        mel_values = (
            torch.zeros(
                (
                    self.sequence_length * MEL_CONFIG.frames_per_position,
                    MEL_CONFIG.num_mel_bins,
                ),
                dtype=torch.float32,
            )
            if self.representation == "mel"
            else None
        )
        mel_mask = (
            torch.zeros(self.sequence_length, dtype=torch.bool)
            if self.representation == "mel"
            else None
        )

        output_offset = 0
        while output_offset < self.sequence_length:
            segment = self.segments[segment_id]
            available = int(segment["logical_length"]) - segment_offset
            take = min(self.sequence_length - output_offset, available)
            output_slice = slice(output_offset, output_offset + take)
            if int(segment["kind"]) == MIDI_KIND:
                assert mel_values is not None and mel_mask is not None
                values = self._render(segment)
                input_ids[output_slice] = int(self.metadata["eos_token_id"])
                source_start = segment_offset * MEL_CONFIG.frames_per_position
                source_end = (segment_offset + take) * MEL_CONFIG.frames_per_position
                output_start = output_offset * MEL_CONFIG.frames_per_position
                output_end = (output_offset + take) * MEL_CONFIG.frames_per_position
                mel_values[output_start:output_end] = values[source_start:source_end]
                mel_mask[output_slice] = True
            else:
                source_start = int(segment["source_offset"]) + segment_offset
                values = self.tokens[source_start : source_start + take]
                input_ids[output_slice] = torch.from_numpy(
                    np.asarray(values, dtype=np.int64)
                )
            output_offset += take
            segment_offset += take
            if segment_offset == int(segment["logical_length"]):
                segment_id += 1
                segment_offset = 0

        return {
            "input_ids": input_ids,
            "mel_values": mel_values,
            "mel_mask": mel_mask,
        }

    def _render(self, segment: np.void) -> torch.Tensor:
        if self.piano_blocks is None:
            raise RuntimeError("prepared stream has no piano blocks")
        if self._renderer is None:
            if self.pianoteq is None:
                raise RuntimeError("Pianoteq executable was not configured")
            self._renderer = PianoteqRenderer(self.pianoteq)
        start = int(segment["source_offset"])
        end = start + int(segment["storage_length"])
        line = bytes(self.piano_blocks[start:end]).decode("utf-8")
        values = self._renderer(line, int(segment["target_samples"]))
        expected = (
            int(segment["logical_length"]) * MEL_CONFIG.frames_per_position,
            MEL_CONFIG.num_mel_bins,
        )
        if values.shape != expected:
            raise RuntimeError(
                f"Expected rendered mel shape {expected}, got {values.shape}"
            )
        return values


def collate_examples(
    examples: list[dict[str, torch.Tensor | None]],
) -> dict[str, torch.Tensor | None]:
    input_ids = torch.stack([example["input_ids"] for example in examples])
    if examples[0]["mel_values"] is None:
        return {"input_ids": input_ids, "mel_values": None, "mel_mask": None}
    return {
        "input_ids": input_ids,
        "mel_values": torch.stack([example["mel_values"] for example in examples]),
        "mel_mask": torch.stack([example["mel_mask"] for example in examples]),
    }


def init_data_worker(_: int) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def mmap_or_empty(path: Path, dtype: np.dtype, length: int) -> np.ndarray:
    if length == 0:
        return np.empty(0, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=(length,))
