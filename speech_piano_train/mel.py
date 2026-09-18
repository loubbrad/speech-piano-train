"""Whisper-style 80-bin log-mel features at a 10 ms frame rate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers.audio_utils import mel_filter_bank


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 16_000
    num_mel_bins: int = 80
    hop_length: int = 160
    n_fft: int = 400
    frames_per_position: int = 4


CONFIG = MelConfig()


class MelEncoder:
    def __init__(self) -> None:
        self.window = torch.hann_window(CONFIG.n_fft, periodic=True)
        filters = mel_filter_bank(
            num_frequency_bins=CONFIG.n_fft // 2 + 1,
            num_mel_filters=CONFIG.num_mel_bins,
            min_frequency=0.0,
            max_frequency=CONFIG.sample_rate / 2,
            sampling_rate=CONFIG.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.filters = torch.from_numpy(
            np.ascontiguousarray(filters.T, dtype=np.float32)
        )

    @torch.inference_mode()
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2 or audio.shape[0] < 1:
            raise ValueError(f"Expected audio [channels, samples], got {audio.shape}")
        mono = audio.float().mean(dim=0)
        if mono.numel() == 0:
            return torch.empty((0, CONFIG.num_mel_bins), dtype=torch.float32)
        spectra = torch.stft(
            mono,
            CONFIG.n_fft,
            hop_length=CONFIG.hop_length,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        magnitudes = spectra[..., :-1].abs().square().contiguous()
        values = (self.filters @ magnitudes).clamp_min(1e-10).log10()
        values = torch.maximum(values, values.max() - 8.0)
        return ((values + 4.0) / 4.0).transpose(0, 1).contiguous()
