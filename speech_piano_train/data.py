from __future__ import annotations

import json
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

TOKEN_DTYPE = np.dtype("<u4")


@dataclass(frozen=True)
class Document:
    youtube_id: str
    modality: str
    path: Path


_WORKER_TOKENIZER: Any = None


def prepare_data(
    data_config: DataConfig,
    model_config: ModelConfig,
    *,
    tokenizer: Any = None,
) -> dict[str, Any]:
    dataset_dir = data_config.dataset_path
    output_dir = data_config.prepared_path
    documents_dir = dataset_dir / "documents"
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
    total_tokens = 0
    with (
        (output_dir / "tokens.bin").open("wb") as token_file,
        (output_dir / "order.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as order_file,
    ):
        iterator = tokenize_documents(
            documents,
            model_config,
            workers=data_config.workers,
            tokenizer=tokenizer,
        )
        for document, token_ids in tqdm(
            iterator,
            total=len(documents),
            desc=f"Tokenizing {data_config.variant}",
            unit="documents",
        ):
            encoded = np.empty(len(token_ids) + 1, dtype=TOKEN_DTYPE)
            encoded[:-1] = token_ids
            encoded[-1] = eos_token_id
            start = total_tokens
            encoded.tofile(token_file)
            total_tokens += len(encoded)
            counts[document.modality] += 1
            order_file.write(
                json.dumps(
                    {
                        "youtube_id": document.youtube_id,
                        "modality": document.modality,
                        "path": document.path.relative_to(documents_dir).as_posix(),
                        "token_start": start,
                        "token_count": len(encoded),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    sequences = total_tokens // data_config.sequence_length
    metadata = {
        "variant": data_config.variant,
        "seed": data_config.seed,
        "sequence_length": data_config.sequence_length,
        "tokenizer": model_config.name,
        "eos_token_id": eos_token_id,
        "documents_by_modality": counts,
        "tokens": total_tokens,
        "sequences": sequences,
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
    if any(item.get("is_benchmark_source") for item in items):
        raise ValueError("Training split contains a benchmark-source item")

    documents: list[Document] = []
    if variant == "interleaved":
        documents = [
            Document(
                youtube_id=item["youtube_id"],
                modality="interleaved",
                path=documents_dir / item["document"],
            )
            for item in items
        ]
    elif variant == "separated":
        for item in items:
            for modality, field in (
                ("speech", "speech_document"),
                ("midi", "midi_document"),
            ):
                path = documents_dir / item[field]
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


def has_adjacent_pair(documents: list[Document]) -> bool:
    return any(
        first.youtube_id == second.youtube_id for first, second in pairwise(documents)
    )


def load_tokenizer(model_config: ModelConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_config.name)
    tokenizer.model_max_length = 2**60
    return tokenizer


def tokenize_documents(
    documents: list[Document],
    model_config: ModelConfig,
    *,
    workers: int,
    tokenizer: Any,
) -> Iterator[tuple[Document, list[int]]]:
    if workers == 1:
        for document in documents:
            yield document, encode_document(document, tokenizer)
        return

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(model_config.name,),
    ) as executor:
        yield from executor.map(_encode_worker, documents, chunksize=4)


def encode_document(document: Document, tokenizer: Any) -> list[int]:
    text = document.path.read_text(encoding="utf-8")
    return tokenizer.encode(text, add_special_tokens=False)


def _init_worker(name: str) -> None:
    global _WORKER_TOKENIZER
    from transformers import AutoTokenizer

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(name)
    _WORKER_TOKENIZER.model_max_length = 2**60


def _encode_worker(document: Document) -> tuple[Document, list[int]]:
    return document, encode_document(document, _WORKER_TOKENIZER)


class TokenDataset(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        self.sequence_length = int(self.metadata["sequence_length"])
        self.tokens = np.memmap(
            self.root / "tokens.bin",
            dtype=TOKEN_DTYPE,
            mode="r",
            shape=(int(self.metadata["tokens"]),),
        )

    def __len__(self) -> int:
        return int(self.metadata["sequences"])

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.sequence_length
        values = self.tokens[start : start + self.sequence_length]
        return torch.from_numpy(np.asarray(values, dtype=np.int64))
