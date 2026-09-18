import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from speech_piano_train.config import DataConfig, ModelConfig
from speech_piano_train.data import (
    TOKEN_DTYPE,
    TokenDataset,
    collate_examples,
    compile_mel_runs,
    has_adjacent_pair,
    ordered_documents,
    prepare_data,
)
from speech_piano_train.prepared_stream import (
    INDEX_DTYPE,
    MIDI_KIND,
    SEGMENT_DTYPE,
    MidiRun,
    PreparedStreamWriter,
    TokenRun,
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


def data_config(
    root: Path,
    variant: str,
    *,
    representation: str = "midi_text",
    sequence_length: int = 4,
) -> DataConfig:
    return DataConfig(
        dataset_dir=root,
        prepared_dir=root / "prepared",
        variant=variant,
        representation=representation,
        seed=7,
        sequence_length=sequence_length,
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
    assert metadata["logical_positions"] == 12
    assert metadata["segments"] == 8
    assert metadata["sequences"] == 3
    assert metadata["representation"] == "midi_text"
    assert metadata["piano_block_bytes"] == 0
    assert len(dataset) == 3
    assert dataset[0]["input_ids"].shape == (4,)
    assert dataset[0]["mel_values"] is None
    raw = np.fromfile(data.prepared_path / "tokens.bin", dtype=TOKEN_DTYPE)
    assert raw.tolist().count(FakeTokenizer.eos_token_id) == 4
    reconstructed = torch.cat(
        [dataset[index]["input_ids"] for index in range(len(dataset))]
    )
    assert reconstructed.tolist() == raw[: len(reconstructed)].tolist()
    assert dataset[-1]["input_ids"].tolist() == dataset[2]["input_ids"].tolist()
    with pytest.raises(IndexError):
        dataset[len(dataset)]

    batch = collate_examples([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (2, 4)
    assert batch["mel_values"] is None
    assert batch["mel_mask"] is None

    assert (data.prepared_path / "tokens.bin").stat().st_size == (
        metadata["tokens"] * TOKEN_DTYPE.itemsize
    )
    assert (data.prepared_path / "segments.bin").stat().st_size == (
        metadata["segments"] * SEGMENT_DTYPE.itemsize
    )
    assert (data.prepared_path / "index.bin").stat().st_size == (
        metadata["sequences"] * INDEX_DTYPE.itemsize
    )

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


def test_audio_compiler_tokenizes_only_contiguous_text_runs() -> None:
    piano = (
        "<piano>unit=10ms;advance=5s;"
        "notes(midi_pitch,velocity,onset,duration):60,80,0,10;</piano>"
    )
    runs = list(
        compile_mel_runs(
            f"before\n{piano}\nafter",
            FakeTokenizer(),
            seed=7,
            document_id="aaaaaaaaaaa",
        )
    )

    assert len(runs) == 3
    assert "".join(map(chr, runs[0].token_ids)) == "before\n<piano>"
    assert isinstance(runs[1], MidiRun)
    assert runs[1].line == piano
    assert 100 * 16 <= runs[1].target_samples <= 1_100 * 16
    assert "".join(map(chr, runs[2].token_ids)) == "</piano>\nafter"


def test_dataset_slices_midi_segments_across_examples(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    line = (
        "<piano>unit=10ms;advance=5s;"
        "notes(midi_pitch,velocity,onset,duration):60,80,0,10;</piano>"
    )
    with PreparedStreamWriter(root, sequence_length=4) as writer:
        writer.append(TokenRun([10, 11]))
        writer.append(MidiRun(line, target_samples=3_840, logical_length=6))
        writer.append(TokenRun([12, 13, 14, 15]))
        counts = writer.close()
    metadata = {
        "schema_version": 1,
        "representation": "mel",
        "sequence_length": 4,
        "eos_token_id": 500,
        **counts,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    calls = []

    def render(rendered_line: str, target_samples: int) -> torch.Tensor:
        calls.append((rendered_line, target_samples))
        return torch.arange(24, dtype=torch.float32)[:, None].expand(24, 80)

    dataset = TokenDataset(root, renderer=render)
    first = dataset[0]
    second = dataset[1]
    third = dataset[2]

    assert first["input_ids"].tolist() == [10, 11, 500, 500]
    assert first["mel_mask"].tolist() == [False, False, True, True]
    assert first["mel_values"][8:, 0].tolist() == list(range(8))
    assert second["mel_mask"].all()
    assert second["mel_values"][:, 0].tolist() == list(range(8, 24))
    assert third["input_ids"].tolist() == [12, 13, 14, 15]
    assert not third["mel_mask"].any()
    assert not third["mel_values"].any()
    assert calls == [(line, 3_840), (line, 3_840)]


def test_builds_and_loads_complete_mel_dataset(tmp_path: Path) -> None:
    youtube_id = "aaaaaaaaaaa"
    folder = tmp_path / "documents" / youtube_id[:2]
    folder.mkdir(parents=True)
    piano_lines = [
        (
            "<piano>unit=10ms;advance=5s;"
            "notes(midi_pitch,velocity,onset,duration):60,80,0,40;</piano>"
        ),
        (
            "<piano>unit=10ms;advance=5s;"
            "notes(midi_pitch,velocity,onset,duration):72,90,0,60;</piano>"
        ),
    ]
    document = f"before\n{piano_lines[0]}\nbetween\n{piano_lines[1]}\nafter"
    (folder / f"{youtube_id}.interleaved.doc.txt").write_text(
        document,
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 3,
                "release": "0.5",
                "items": [{"youtube_id": youtube_id, "split": "train"}],
            }
        ),
        encoding="utf-8",
    )
    text_config = data_config(
        tmp_path,
        "interleaved",
        representation="midi_text",
        sequence_length=8,
    )
    text_metadata = prepare_data(
        text_config,
        model_config(),
        tokenizer=FakeTokenizer(),
    )
    text_dataset = TokenDataset(text_config.prepared_path)
    expected_text_tokens = [*map(ord, document), FakeTokenizer.eos_token_id]
    loaded_text_tokens = torch.cat(
        [text_dataset[index]["input_ids"] for index in range(len(text_dataset))]
    )
    assert text_metadata["representation"] == "midi_text"
    assert text_metadata["piano_block_bytes"] == 0
    assert loaded_text_tokens.tolist() == expected_text_tokens[: len(loaded_text_tokens)]

    config = data_config(
        tmp_path,
        "interleaved",
        representation="mel",
        sequence_length=8,
    )

    metadata = prepare_data(config, model_config(), tokenizer=FakeTokenizer())
    tokens = np.fromfile(config.prepared_path / "tokens.bin", dtype=TOKEN_DTYPE)
    segments = np.fromfile(
        config.prepared_path / "segments.bin",
        dtype=SEGMENT_DTYPE,
    )
    piano_bytes = (config.prepared_path / "piano_blocks.bin").read_bytes()

    assert metadata["representation"] == "mel"
    assert metadata["documents_by_modality"] == {
        "interleaved": 1,
        "speech": 0,
        "midi": 0,
    }
    assert metadata["segments"] == 6
    assert [int(segment["kind"]) for segment in segments] == [0, 1, 0, 1, 0, 0]
    assert piano_bytes == "".join(piano_lines).encode()
    assert metadata["piano_block_bytes"] == len(piano_bytes)
    assert metadata["logical_positions"] == sum(
        int(segment["logical_length"]) for segment in segments
    )
    assert metadata["sequences"] == metadata["logical_positions"] // 8

    order = json.loads((config.prepared_path / "order.jsonl").read_text())
    assert order["logical_start"] == 0
    assert order["logical_count"] == metadata["logical_positions"]
    assert order["token_start"] == 0
    assert order["token_count"] == metadata["tokens"]

    render_calls: list[tuple[str, int]] = []

    def render(line: str, target_samples: int) -> torch.Tensor:
        render_calls.append((line, target_samples))
        block = next(segment for segment in segments if _segment_line(segment) == line)
        frames = int(block["logical_length"]) * 4
        base = 1_000 if "60,80" in line else 2_000
        return (base + torch.arange(frames, dtype=torch.float32))[:, None].expand(
            frames, 80
        )

    def _segment_line(segment: np.void) -> str | None:
        if int(segment["kind"]) != MIDI_KIND:
            return None
        start = int(segment["source_offset"])
        end = start + int(segment["storage_length"])
        return piano_bytes[start:end].decode()

    expected_ids: list[int] = []
    expected_mask: list[bool] = []
    expected_mel: list[float] = []
    for segment in segments:
        logical_length = int(segment["logical_length"])
        if int(segment["kind"]) == MIDI_KIND:
            line = _segment_line(segment)
            assert line is not None
            base = 1_000 if "60,80" in line else 2_000
            expected_ids.extend([FakeTokenizer.eos_token_id] * logical_length)
            expected_mask.extend([True] * logical_length)
            expected_mel.extend(
                (base + torch.arange(logical_length * 4)).tolist()
            )
        else:
            start = int(segment["source_offset"])
            end = start + logical_length
            expected_ids.extend(tokens[start:end].tolist())
            expected_mask.extend([False] * logical_length)
            expected_mel.extend([0.0] * (logical_length * 4))

    dataset = TokenDataset(config.prepared_path, renderer=render)
    examples = [dataset[index] for index in range(len(dataset))]
    used_positions = len(dataset) * config.sequence_length
    loaded_ids = torch.cat([example["input_ids"] for example in examples])
    loaded_mask = torch.cat([example["mel_mask"] for example in examples])
    loaded_mel = torch.cat([example["mel_values"] for example in examples])

    assert loaded_ids.tolist() == expected_ids[:used_positions]
    assert loaded_mask.tolist() == expected_mask[:used_positions]
    assert loaded_mel[:, 0].tolist() == expected_mel[: used_positions * 4]
    assert {line for line, _ in render_calls} == set(piano_lines)
    assert all(target_samples > 0 for _, target_samples in render_calls)

    batch = collate_examples(examples[:2])
    assert batch["input_ids"].shape == (2, 8)
    assert batch["mel_mask"].shape == (2, 8)
    assert batch["mel_values"].shape == (2, 32, 80)
