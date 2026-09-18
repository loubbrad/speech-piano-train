import base64
import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from speech_piano_train.codec import (
    CONFIG,
    codec_tokens_from_bytes,
    codec_tokens_to_bytes,
    decode_wav,
    encode_wav,
)


class _LocalCodec(nn.Module):
    """Small exact model with the same temporal receptive fields."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        frames = audio[:, :1].reshape(audio.shape[0], 1, -1, CONFIG.hop_samples)
        frames = frames[..., 0]
        values = F.conv1d(F.pad(frames, (14, 2)), torch.ones(1, 1, 17))
        return values.long().repeat(1, CONFIG.num_codebooks, 1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        frames = codes[:, :1].float()
        values = F.conv1d(F.pad(frames, (14, 1)), torch.ones(1, 1, 16))
        values = values[..., :-1].repeat_interleave(CONFIG.hop_samples, -1)
        return values.repeat(1, CONFIG.channels, 1)


def test_codec_token_file_roundtrip() -> None:
    codes = torch.randint(0, CONFIG.codebook_size, (CONFIG.num_codebooks, 17))
    payload = codec_tokens_to_bytes(codes, 12_345)

    decoded, num_samples = codec_tokens_from_bytes(payload)

    assert torch.equal(decoded, codes)
    assert num_samples == 12_345
    value = json.loads(payload)
    assert set(value) == {"schema_version", "num_samples", "codes_b64"}
    assert len(base64.b64decode(value["codes_b64"])) == 17 * 15


def test_codec_token_file_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="12"):
        codec_tokens_to_bytes(torch.zeros((11, 4), dtype=torch.long), 1)


def test_codec_token_file_rejects_bad_code() -> None:
    codes = torch.zeros((CONFIG.num_codebooks, 1), dtype=torch.long)
    codes[0, 0] = CONFIG.codebook_size
    with pytest.raises(ValueError, match="1023"):
        codec_tokens_to_bytes(codes, 1)


def test_halo_batching_matches_whole_file() -> None:
    codec = _LocalCodec()
    audio = torch.zeros((2, 24 * CONFIG.hop_samples + 31))
    audio[:, :: CONFIG.hop_samples] = torch.arange(25) % 3

    whole_codes = encode_wav(codec, audio, core_frames=100, batch_size=1)
    batched_codes = encode_wav(codec, audio, core_frames=3, batch_size=4)
    assert torch.equal(batched_codes, whole_codes)

    whole_audio = decode_wav(codec, whole_codes, core_frames=100, batch_size=1)
    batched_audio = decode_wav(codec, whole_codes, core_frames=3, batch_size=4)
    assert torch.equal(batched_audio, whole_audio)
