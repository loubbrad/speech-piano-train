"""Online Pianoteq rendering for prepared MIDI-backed audio segments."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torchcodec.decoders import AudioDecoder

from speech_piano_train.mel import CONFIG, MelEncoder
from speech_piano_train.midi import MidiTextTokenizer

PRESET = "NY Steinway D Classical Recording"


def resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path("."):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    else:
        resolved = shutil.which(value)
        if resolved is not None:
            return resolved
    raise FileNotFoundError(f"Could not find executable Pianoteq at {value!r}")


def temporary_root() -> Path | None:
    shared_memory = Path("/dev/shm")
    if shared_memory.is_dir() and os.access(shared_memory, os.W_OK):
        return shared_memory
    return None


class PianoteqRenderer:
    def __init__(self, executable: str) -> None:
        self.executable = executable
        self.midi_tokenizer = MidiTextTokenizer()
        self.mel_encoder = MelEncoder()

    def __call__(self, line: str, target_samples: int) -> torch.Tensor:
        midi = self.midi_tokenizer.detokenize(line)
        with tempfile.TemporaryDirectory(
            prefix="speech-piano-",
            dir=temporary_root(),
        ) as temporary:
            root = Path(temporary)
            midi_path = root / "segment.mid"
            wav_path = root / "segment.wav"
            midi.to_midi().save(midi_path)
            completed = subprocess.run(
                [
                    self.executable,
                    "--headless",
                    "--preset",
                    PRESET,
                    "--midi",
                    str(midi_path),
                    "--wav",
                    str(wav_path),
                    "--rate",
                    str(CONFIG.sample_rate),
                    "--bit-depth",
                    "24",
                    "--dither",
                    "OFF",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                details = completed.stdout[-4_000:].strip()
                raise RuntimeError(
                    f"Pianoteq exited with status {completed.returncode}: {details}"
                )
            if not wav_path.is_file() or wav_path.stat().st_size == 0:
                raise RuntimeError("Pianoteq did not create a non-empty WAV file")
            audio = (
                AudioDecoder(wav_path, sample_rate=CONFIG.sample_rate)
                .get_all_samples()
                .data
            )

        if audio.shape[-1] < target_samples:
            audio = F.pad(audio, (0, target_samples - audio.shape[-1]))
        else:
            audio = audio[..., :target_samples]
        values = self.mel_encoder(audio)
        remainder = values.shape[0] % CONFIG.frames_per_position
        if remainder:
            values = F.pad(values, (0, 0, 0, CONFIG.frames_per_position - remainder))
        return values
