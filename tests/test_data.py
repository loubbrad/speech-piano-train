import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from speech_piano_train.config import DataConfig, ModelConfig
from speech_piano_train.data import (
    TOKEN_DTYPE,
    TokenDataset,
    has_adjacent_pair,
    ordered_documents,
    prepare_data,
)


class FakeTokenizer:
    eos_token_id = 500

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def write_upstream_fixture(root: Path) -> dict:
    documents = root / "documents"
    items = []
    for youtube_id, split in (
        ("aaaaaaaaaaa", "train"),
        ("bbbbbbbbbbb", "train"),
        ("ccccccccccc", "validation"),
    ):
        prefix = youtube_id[:2]
        folder = documents / prefix
        folder.mkdir(parents=True, exist_ok=True)
        paths = {
            "document": f"{prefix}/{youtube_id}.interleaved.doc.txt",
            "speech_document": f"{prefix}/{youtube_id}.speech.doc.txt",
            "midi_document": f"{prefix}/{youtube_id}.midi.doc.txt",
        }
        (documents / paths["document"]).write_text(
            f"I{youtube_id[-1]}", encoding="utf-8"
        )
        (documents / paths["speech_document"]).write_text(
            f"S{youtube_id[-1]}", encoding="utf-8"
        )
        (documents / paths["midi_document"]).write_text(
            f"M{youtube_id[-1]}", encoding="utf-8"
        )
        items.append(
            {
                "youtube_id": youtube_id,
                "split": split,
                "is_benchmark_source": False,
                **paths,
            }
        )
    manifest = {
        "manifest_version": 3,
        "source": {"caption_source": "youtube"},
        "items": items,
    }
    (documents / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def write_v05_fixture(root: Path) -> dict:
    documents = root / "documents"
    items = []
    for youtube_id, split in (
        ("aaaaaaaaaaa", "train"),
        ("bbbbbbbbbbb", "train"),
        ("ccccccccccc", "validation"),
    ):
        folder = documents / youtube_id[:2]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{youtube_id}.interleaved.doc.txt").write_text(
            f"I{youtube_id[-1]}", encoding="utf-8"
        )
        (folder / f"{youtube_id}.deinterleaved-speech.doc.txt").write_text(
            f"D{youtube_id[-1]}", encoding="utf-8"
        )
        (folder / f"{youtube_id}.deinterleaved-midi.doc.txt").write_text(
            f"P{youtube_id[-1]}", encoding="utf-8"
        )
        # These deliberately differ so the test catches use of the old files.
        (folder / f"{youtube_id}.speech.doc.txt").write_text(
            f"S{youtube_id[-1]}", encoding="utf-8"
        )
        (folder / f"{youtube_id}.midi.doc.txt").write_text(
            f"M{youtube_id[-1]}", encoding="utf-8"
        )
        items.append({"youtube_id": youtube_id, "split": split})

    manifest = {
        "manifest_version": 3,
        "release": "0.5",
        "splits": {
            "benchmark": {"source_youtube_ids": ["ccccccccccc"]},
        },
        "items": items,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def data_config(root: Path, variant: str) -> DataConfig:
    return DataConfig(
        dataset_dir=root,
        prepared_dir=root / "prepared",
        variant=variant,
        seed=7,
        sequence_length=4,
        workers=1,
    )


def model_config() -> ModelConfig:
    return ModelConfig(name="test/tokenizer")


def test_separated_order_excludes_held_out_and_adjacent_pairs(
    tmp_path: Path,
) -> None:
    manifest = write_upstream_fixture(tmp_path)

    documents = ordered_documents(
        tmp_path / "documents",
        manifest,
        variant="separated",
        seed=7,
    )

    assert len(documents) == 4
    assert {document.youtube_id for document in documents} == {
        "aaaaaaaaaaa",
        "bbbbbbbbbbb",
    }
    assert not has_adjacent_pair(documents)


def test_builds_deterministic_memory_mapped_data(tmp_path: Path) -> None:
    write_upstream_fixture(tmp_path)
    data = data_config(tmp_path, "separated")
    model = model_config()

    metadata = prepare_data(data, model, tokenizer=FakeTokenizer())
    dataset = TokenDataset(data.prepared_path)

    assert metadata["documents_by_modality"] == {
        "interleaved": 0,
        "speech": 2,
        "midi": 2,
    }
    assert metadata["tokens"] == 12
    assert metadata["sequences"] == 3
    assert len(dataset) == 3
    assert dataset[0].shape == (4,)
    raw = np.fromfile(data.prepared_path / "tokens.bin", dtype=TOKEN_DTYPE)
    assert raw.tolist().count(FakeTokenizer.eos_token_id) == 4

    order = [
        json.loads(line)
        for line in (data.prepared_path / "order.jsonl").read_text().splitlines()
    ]
    assert all(
        first["youtube_id"] != second["youtube_id"] for first, second in pairwise(order)
    )

    with pytest.raises(FileExistsError):
        prepare_data(data, model, tokenizer=FakeTokenizer())


@pytest.mark.parametrize(
    ("variant", "expected_suffixes"),
    [
        ("interleaved", {".interleaved.doc.txt"}),
        (
            "separated",
            {
                ".deinterleaved-speech.doc.txt",
                ".deinterleaved-midi.doc.txt",
            },
        ),
    ],
)
def test_v05_manifest_and_document_layout(
    tmp_path: Path,
    variant: str,
    expected_suffixes: set[str],
) -> None:
    manifest = write_v05_fixture(tmp_path)

    documents = ordered_documents(
        tmp_path / "documents",
        manifest,
        variant=variant,
        seed=7,
    )

    assert {document.youtube_id for document in documents} == {
        "aaaaaaaaaaa",
        "bbbbbbbbbbb",
    }
    assert all(
        any(document.path.name.endswith(suffix) for suffix in expected_suffixes)
        for document in documents
    )

    metadata = prepare_data(
        data_config(tmp_path, variant),
        model_config(),
        tokenizer=FakeTokenizer(),
    )
    expected_counts = (
        {"interleaved": 2, "speech": 0, "midi": 0}
        if variant == "interleaved"
        else {"interleaved": 0, "speech": 2, "midi": 2}
    )
    assert metadata["documents_by_modality"] == expected_counts

    order_paths = [
        json.loads(line)["path"]
        for line in (data_config(tmp_path, variant).prepared_path / "order.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(
        any(path.endswith(suffix) for suffix in expected_suffixes)
        for path in order_paths
    )


def test_v05_benchmark_source_cannot_enter_training(tmp_path: Path) -> None:
    manifest = write_v05_fixture(tmp_path)
    manifest["splits"]["benchmark"]["source_youtube_ids"] = ["aaaaaaaaaaa"]

    with pytest.raises(ValueError, match="benchmark-source"):
        ordered_documents(
            tmp_path / "documents",
            manifest,
            variant="interleaved",
            seed=7,
        )


def test_empty_midi_document_is_skipped(tmp_path: Path) -> None:
    manifest = write_upstream_fixture(tmp_path)
    midi = tmp_path / "documents/aa/aaaaaaaaaaa.midi.doc.txt"
    midi.write_text("", encoding="utf-8")

    documents = ordered_documents(
        tmp_path / "documents",
        manifest,
        variant="separated",
        seed=7,
    )

    assert len(documents) == 3
    assert not any(document.path == midi for document in documents)
