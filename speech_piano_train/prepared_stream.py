"""Compact indexed logical streams containing token and MIDI runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

TOKEN_KIND = 0
MIDI_KIND = 1
TOKEN_DTYPE = np.dtype("<u4")
SEGMENT_DTYPE = np.dtype(
    [
        ("kind", "u1"),
        ("reserved", "u1", (7,)),
        ("source_offset", "<u8"),
        ("storage_length", "<u8"),
        ("logical_length", "<u8"),
        ("target_samples", "<u8"),
    ]
)
INDEX_DTYPE = np.dtype([("segment_id", "<u8"), ("offset", "<u8")])


@dataclass(frozen=True)
class TokenRun:
    token_ids: list[int]


@dataclass(frozen=True)
class MidiRun:
    line: str
    target_samples: int
    logical_length: int


class PreparedStreamWriter:
    def __init__(self, root: Path, sequence_length: int) -> None:
        self.root = root
        self.sequence_length = sequence_length
        self.token_file = (root / "tokens.bin").open("wb")
        self.piano_file = (root / "piano_blocks.bin").open("wb")
        self.segment_file = (root / "segments.bin").open("wb")
        self.index_file = (root / "index.bin").open("wb")
        self.tokens = 0
        self.piano_bytes = 0
        self.segments = 0
        self.logical_positions = 0
        self._next_example_start = 0

    def append(self, run: TokenRun | MidiRun) -> None:
        if isinstance(run, TokenRun):
            self._append_tokens(run)
        else:
            self._append_midi(run)

    def _append_tokens(self, run: TokenRun) -> None:
        if not run.token_ids:
            return
        values = np.asarray(run.token_ids, dtype=TOKEN_DTYPE)
        source_offset = self.tokens
        values.tofile(self.token_file)
        self.tokens += len(values)
        self._append_segment(
            TOKEN_KIND,
            source_offset,
            len(values),
            len(values),
            0,
        )

    def _append_midi(self, run: MidiRun) -> None:
        payload = run.line.encode("utf-8")
        source_offset = self.piano_bytes
        self.piano_file.write(payload)
        self.piano_bytes += len(payload)
        self._append_segment(
            MIDI_KIND,
            source_offset,
            len(payload),
            run.logical_length,
            run.target_samples,
        )

    def _append_segment(
        self,
        kind: int,
        source_offset: int,
        storage_length: int,
        logical_length: int,
        target_samples: int,
    ) -> None:
        if logical_length <= 0:
            return
        start = self.logical_positions
        end = start + logical_length
        while self._next_example_start < end:
            cursor = np.array(
                [(self.segments, self._next_example_start - start)],
                dtype=INDEX_DTYPE,
            )
            cursor.tofile(self.index_file)
            self._next_example_start += self.sequence_length
        segment = np.zeros(1, dtype=SEGMENT_DTYPE)
        segment["kind"] = kind
        segment["source_offset"] = source_offset
        segment["storage_length"] = storage_length
        segment["logical_length"] = logical_length
        segment["target_samples"] = target_samples
        segment.tofile(self.segment_file)
        self.segments += 1
        self.logical_positions = end

    def close(self) -> dict[str, int]:
        sequences = self.logical_positions // self.sequence_length
        self.index_file.flush()
        self.index_file.truncate(sequences * INDEX_DTYPE.itemsize)
        for file in (
            self.token_file,
            self.piano_file,
            self.segment_file,
            self.index_file,
        ):
            file.close()
        return {
            "tokens": self.tokens,
            "piano_block_bytes": self.piano_bytes,
            "segments": self.segments,
            "logical_positions": self.logical_positions,
            "sequences": sequences,
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        if not self.token_file.closed:
            self.close()
