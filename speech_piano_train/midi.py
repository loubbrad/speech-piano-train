"""Parse the corpus' text-compatible piano format and convert it with Aria."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

PIANO_LINE_RE = re.compile(
    r"^<piano>unit=(?P<unit_ms>\d+)ms;advance=(?P<advance_s>\d+)s;"
    r"notes\(midi_pitch,velocity,onset,duration\):"
    r"(?P<records>.*)</piano>$"
)


@dataclass(frozen=True)
class PianoBlock:
    line: str
    unit_ms: int
    advance_ms: int
    records: tuple[str, ...]
    end_ms: int


def parse_piano_line(line: str) -> PianoBlock:
    match = PIANO_LINE_RE.fullmatch(line)
    if match is None:
        raise ValueError(f"invalid serialized piano block: {line[:160]!r}")
    unit_ms = int(match.group("unit_ms"))
    advance_ms = int(match.group("advance_s")) * 1_000
    if unit_ms <= 0 or advance_ms <= 0:
        raise ValueError("piano time units must be positive")

    records = tuple(value for value in match.group("records").split(";") if value)
    if not records:
        raise ValueError("piano block contains no records")
    cursor_ms = 0
    end_ms = 0
    note_count = 0
    last_was_advance = False
    for record in records:
        if record == "advance":
            if note_count == 0:
                raise ValueError("piano block begins with an advance")
            cursor_ms += advance_ms
            last_was_advance = True
            continue
        fields = record.split(",")
        if len(fields) != 4:
            raise ValueError(f"invalid piano note record: {record!r}")
        try:
            pitch, velocity, onset, duration = map(int, fields)
        except ValueError as error:
            raise ValueError(f"non-integer piano note record: {record!r}") from error
        if not 0 <= pitch <= 127 or not 0 <= velocity <= 127:
            raise ValueError(f"piano pitch or velocity is out of range: {record!r}")
        if onset < 0 or onset * unit_ms >= advance_ms or duration <= 0:
            raise ValueError(f"invalid piano timing: {record!r}")
        end_ms = max(end_ms, cursor_ms + (onset + duration) * unit_ms)
        note_count += 1
        last_was_advance = False
    if note_count == 0:
        raise ValueError("piano block contains no notes")
    if last_was_advance:
        raise ValueError("piano block ends with an advance")
    return PianoBlock(line, unit_ms, advance_ms, records, end_ms)


class MidiTextTokenizer:
    """Bidirectional adapter between serialized piano lines and Aria MIDI."""

    def __init__(self) -> None:
        from ariautils.tokenizer import AbsTokenizer

        self._tokenizer = AbsTokenizer()
        if self._tokenizer.include_pedal:
            raise RuntimeError("AbsTokenizer unexpectedly includes pedal tokens")

    def detokenize(self, line: str) -> Any:
        block = parse_piano_line(line)
        return self._tokenizer.detokenize(self._aria_tokens(block))

    def tokenize(self, midi: Any) -> list[Any]:
        return self._tokenizer.tokenize(
            midi,
            remove_preceding_silence=True,
            add_dim_tok=False,
            add_eos_tok=True,
        )

    def validate(self, line: str) -> None:
        block = parse_piano_line(line)
        tokens = self._aria_tokens(block)
        if self.tokenize(self._tokenizer.detokenize(tokens)) != tokens:
            raise ValueError("serialized piano block does not round-trip through Aria")

    def _aria_tokens(self, block: PianoBlock) -> list[Any]:
        tokenizer = self._tokenizer
        if block.unit_ms != int(tokenizer.config["time_step_ms"]):
            raise ValueError(
                f"piano unit is {block.unit_ms}ms; tokenizer expects "
                f"{tokenizer.config['time_step_ms']}ms"
            )
        if block.advance_ms != int(tokenizer.abs_time_step_ms):
            raise ValueError(
                f"piano advance is {block.advance_ms}ms; tokenizer expects "
                f"{tokenizer.abs_time_step_ms}ms"
            )
        tokens: list[Any] = [("prefix", "instrument", "piano"), tokenizer.bos_tok]
        for record in block.records:
            if record == "advance":
                tokens.append(tokenizer.time_tok)
                continue
            pitch, velocity, onset, duration = map(int, record.split(","))
            tokens.extend(
                [
                    ("piano", pitch, velocity),
                    ("onset", onset * block.unit_ms),
                    ("dur", duration * block.unit_ms),
                ]
            )
        tokens.append(tokenizer.eos_tok)
        return tokens
