"""Inkling-compatible discretized mel (dMel) audio representation."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchcodec.decoders import AudioDecoder
from transformers.audio_utils import mel_filter_bank


@dataclass(frozen=True)
class DMelConfig:
    sample_rate: int = 16_000
    num_mel_bins: int = 80
    hop_length: int = 800
    window_length: int = 1_600
    n_fft: int = 1_600
    num_bins: int = 16
    min_value: float = -7.0
    max_value: float = 2.0


CONFIG = DMelConfig()
FILE_SCHEMA = 1
REPRESENTATION = "inkling_dmel"
PACKING = "uint4-msn-first"


class DMelEncoder:
    """Stateless Inkling dMel transform with cached analysis tensors."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.window = torch.hann_window(
            CONFIG.window_length, periodic=True, dtype=torch.float32,
            device=self.device,
        )
        filters = mel_filter_bank(
            num_frequency_bins=CONFIG.n_fft // 2 + 1,
            num_mel_filters=CONFIG.num_mel_bins,
            min_frequency=0.0,
            max_frequency=CONFIG.sample_rate / 2.0,
            sampling_rate=CONFIG.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.mel_filters = torch.from_numpy(
            np.ascontiguousarray(filters.T, dtype=np.float32)
        ).to(self.device)
        self.bin_centers = torch.linspace(
            CONFIG.min_value,
            CONFIG.max_value,
            CONFIG.num_bins,
            dtype=torch.float64,
            device=self.device,
        )

    def _encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        spectra = torch.stft(
            windows.to(self.device),
            CONFIG.n_fft,
            hop_length=CONFIG.hop_length,
            win_length=CONFIG.window_length,
            window=self.window,
            center=False,
            return_complex=True,
        ).abs()
        log_mel = (self.mel_filters @ spectra).clamp_min(1e-10).log10()
        log_mel = log_mel.transpose(1, 2).to(torch.float64)
        return (
            (log_mel.clamp(CONFIG.min_value, CONFIG.max_value).unsqueeze(-1)
             - self.bin_centers)
            .abs()
            .argmin(dim=-1)
            .to(torch.uint8)
        )


@torch.inference_mode()
def encode_wav(
    encoder: DMelEncoder,
    audio: torch.Tensor,
    core_frames: int = 250,
    batch_size: int = 8,
) -> torch.Tensor:
    """Encode a waveform in globally aligned, haloed STFT windows."""
    if core_frames < 1 or batch_size < 1:
        raise ValueError("core_frames and batch_size must be positive")
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or audio.shape[0] < 1:
        raise ValueError(f"Expected audio [channels, samples], got {audio.shape}")
    if not audio.is_floating_point():
        raise TypeError("Audio must be floating point")

    mono = audio.float().mean(dim=0)
    num_samples = mono.numel()
    num_frames = math.ceil(num_samples / CONFIG.hop_length)
    if num_frames == 0:
        return torch.empty((0, CONFIG.num_mel_bins), dtype=torch.uint8)

    # This is Inkling's exact boundary convention: one hop of zero context on
    # the left, and enough zeros on the right to end on a hop boundary.
    right_pad = num_frames * CONFIG.hop_length - num_samples
    padded = F.pad(mono, (CONFIG.n_fft - CONFIG.hop_length, right_pad))

    windows: list[torch.Tensor] = []
    lengths: list[int] = []
    for start in range(0, num_frames, core_frames):
        length = min(core_frames, num_frames - start)
        sample_start = start * CONFIG.hop_length
        sample_end = sample_start + (length - 1) * CONFIG.hop_length + CONFIG.n_fft
        windows.append(padded[sample_start:sample_end])
        lengths.append(length)

    output: list[torch.Tensor] = []
    for start in range(0, len(windows), batch_size):
        selected = windows[start : start + batch_size]
        # Only the final window can be shorter, so split it from full windows.
        buckets: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for index, window in enumerate(selected):
            buckets.setdefault(window.numel(), []).append((index, window))
        batch_output: list[torch.Tensor | None] = [None] * len(selected)
        for bucket in buckets.values():
            values = encoder._encode_windows(torch.stack([item[1] for item in bucket]))
            for (index, _), value in zip(bucket, values, strict=True):
                batch_output[index] = value.cpu()
        output.extend(value for value in batch_output if value is not None)

    result = torch.cat(output, dim=0)
    if result.shape != (num_frames, CONFIG.num_mel_bins):
        raise RuntimeError(f"Unexpected dMel shape {result.shape}")
    return result


def dmel_to_bytes(values: torch.Tensor, num_samples: int) -> bytes:
    """Serialize frame-major dMel values as base64-encoded packed nibbles."""
    _check_values(values)
    if not isinstance(num_samples, int) or num_samples < 0:
        raise ValueError("num_samples must be a nonnegative integer")
    expected_frames = math.ceil(num_samples / CONFIG.hop_length)
    if values.shape[0] != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} frames for {num_samples} samples, "
            f"got {values.shape[0]}"
        )
    flat = values.contiguous().cpu().numpy().astype(np.uint8, copy=False).reshape(-1)
    packed = (flat[0::2] << 4) | flat[1::2]
    return json.dumps(
        {
            "schema_version": FILE_SCHEMA,
            "representation": REPRESENTATION,
            "sample_rate": CONFIG.sample_rate,
            "num_samples": num_samples,
            "shape": [values.shape[0], CONFIG.num_mel_bins],
            "packing": PACKING,
            "dmel_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
        },
        separators=(",", ":"),
    ).encode()


def dmel_from_bytes(payload: bytes) -> tuple[torch.Tensor, int]:
    """Deserialize and validate the Inkling dMel schema."""
    value = json.loads(payload)
    if value.get("schema_version") != FILE_SCHEMA:
        raise ValueError("Unsupported dMel schema")
    if value.get("representation") != REPRESENTATION:
        raise ValueError("Unsupported dMel representation")
    if value.get("sample_rate") != CONFIG.sample_rate:
        raise ValueError("Unsupported dMel sample rate")
    if value.get("packing") != PACKING:
        raise ValueError("Unsupported dMel packing")
    num_samples = value.get("num_samples")
    if not isinstance(num_samples, int) or num_samples < 0:
        raise ValueError("Invalid num_samples")
    shape = value.get("shape")
    expected_frames = math.ceil(num_samples / CONFIG.hop_length)
    if shape != [expected_frames, CONFIG.num_mel_bins]:
        raise ValueError("Invalid dMel shape")
    encoded = value.get("dmel_b64")
    if not isinstance(encoded, str):
        raise TypeError("dMel payload must be a base64 string")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != expected_frames * CONFIG.num_mel_bins // 2:
        raise ValueError("Invalid dMel payload length")
    packed = np.frombuffer(raw, dtype=np.uint8)
    flat = np.empty(packed.size * 2, dtype=np.uint8)
    flat[0::2] = packed >> 4
    flat[1::2] = packed & 0x0F
    values = torch.from_numpy(flat).reshape(expected_frames, CONFIG.num_mel_bins)
    _check_values(values)
    return values, num_samples


def _check_values(values: torch.Tensor) -> None:
    if values.ndim != 2 or values.shape[1] != CONFIG.num_mel_bins:
        raise ValueError(
            f"Expected dMel values [frames, {CONFIG.num_mel_bins}], got {values.shape}"
        )
    if values.is_floating_point():
        raise TypeError("dMel values must be integers")
    if values.numel() and (values.min() < 0 or values.max() >= CONFIG.num_bins):
        raise ValueError("dMel values must be in [0, 15]")


def save_dmel(path: str | Path, values: torch.Tensor, num_samples: int) -> None:
    Path(path).write_bytes(dmel_to_bytes(values, num_samples))


def load_dmel(path: str | Path) -> tuple[torch.Tensor, int]:
    return dmel_from_bytes(Path(path).read_bytes())


def encode_file(
    encoder: DMelEncoder,
    audio_path: str | Path,
    output_path: str | Path,
    core_frames: int = 250,
    batch_size: int = 8,
) -> None:
    decoder = AudioDecoder(audio_path, sample_rate=CONFIG.sample_rate)
    audio = decoder.get_all_samples().data
    values = encode_wav(encoder, audio, core_frames, batch_size)
    save_dmel(output_path, values, audio.shape[-1])
