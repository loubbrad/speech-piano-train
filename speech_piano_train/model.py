"""Qwen model with a small randomly initialized log-mel input adapter."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3_5ForCausalLM
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)

from speech_piano_train.mel import CONFIG

_AUDIO_MODEL_CLASS = "AudioQwen3_5ForCausalLM"
if get_checkpoint_conversion_mapping(_AUDIO_MODEL_CLASS) is None:
    qwen_conversion = get_checkpoint_conversion_mapping("qwen3_5_text")
    if qwen_conversion is None:
        raise RuntimeError("Transformers has no Qwen 3.5 checkpoint conversion")
    register_checkpoint_conversion_mapping(_AUDIO_MODEL_CLASS, qwen_conversion)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        variance = values.float().square().mean(dim=-1, keepdim=True)
        return self.weight * (values * torch.rsqrt(variance + self.eps)).to(dtype)


class MelAdapter(nn.Module):
    def __init__(self, hidden_size: int, eps: float, initializer_range: float) -> None:
        super().__init__()
        width = 512
        self.conv1 = nn.Conv1d(CONFIG.num_mel_bins, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            stride=CONFIG.frames_per_position,
            padding=1,
        )
        self.projection = nn.Linear(width, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size, eps)
        for module in (self.conv1, self.conv2, self.projection):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.gelu(self.conv1(values.transpose(1, 2)))
        values = F.gelu(self.conv2(values)).transpose(1, 2)
        return self.norm(self.projection(values))


class AudioQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.mel_adapter = MelAdapter(
            config.hidden_size,
            config.rms_norm_eps,
            config.initializer_range,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        mel_values: torch.Tensor,
        mel_mask: torch.Tensor,
        **kwargs,
    ):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        mel_embeds = self.mel_adapter(mel_values)
        if mel_embeds.shape[:2] != mel_mask.shape:
            raise ValueError("mel stem output does not match the logical sequence")
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[mel_mask] = mel_embeds[mel_mask].to(inputs_embeds.dtype)
        return super().forward(input_ids=None, inputs_embeds=inputs_embeds, **kwargs)
