import base64
import json

import pytest
import torch

from speech_piano_train.dmel import (
    CONFIG,
    DMelEncoder,
    dmel_from_bytes,
    dmel_to_bytes,
    encode_wav,
)


def test_nibble_file_roundtrip_and_layout() -> None:
    values = (torch.arange(80) % 16).reshape(1, 80).to(torch.uint8)
    payload = dmel_to_bytes(values, 1)
    decoded, num_samples = dmel_from_bytes(payload)

    assert torch.equal(decoded, values)
    assert num_samples == 1
    document = json.loads(payload)
    assert document["packing"] == "uint4-msn-first"
    raw = base64.b64decode(document["dmel_b64"])
    assert raw[:2] == bytes((0x01, 0x23))
    assert len(raw) == 40


def test_file_rejects_bad_values_and_length() -> None:
    values = torch.zeros((1, 80), dtype=torch.int64)
    values[0, 0] = 16
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        dmel_to_bytes(values, 1)

    document = json.loads(dmel_to_bytes(torch.zeros((1, 80), dtype=torch.uint8), 1))
    document["dmel_b64"] = base64.b64encode(b"short").decode()
    with pytest.raises(ValueError, match="payload length"):
        dmel_from_bytes(json.dumps(document).encode())


def test_chunking_matches_whole_audio() -> None:
    generator = torch.Generator().manual_seed(123)
    audio = torch.randn((2, 19_337), generator=generator)
    encoder = DMelEncoder()

    whole = encode_wav(encoder, audio, core_frames=10_000, batch_size=1)
    chunked = encode_wav(encoder, audio, core_frames=3, batch_size=4)

    assert torch.equal(chunked, whole)
    assert chunked.shape == (25, CONFIG.num_mel_bins)


def test_matches_transformers_inkling_processor() -> None:
    from transformers.models.inkling.feature_extraction_inkling import (
        InklingFeatureExtractor,
    )
    generator = torch.Generator().manual_seed(456)
    audio = torch.randn(12_345, generator=generator)
    expected_features = InklingFeatureExtractor()(
        audio.numpy(), sampling_rate=16_000, return_tensors="pt"
    )["input_features"]
    centers = torch.linspace(-7.0, 2.0, 16, dtype=torch.float64)
    expected = (
        (expected_features.double().clamp(-7.0, 2.0).unsqueeze(-1) - centers)
        .abs()
        .argmin(-1)
        .to(torch.uint8)[0]
    )

    actual = encode_wav(DMelEncoder(), audio, core_frames=4, batch_size=2)
    assert torch.equal(actual, expected)


def test_empty_audio() -> None:
    values = encode_wav(DMelEncoder(), torch.empty((1, 0)))
    assert values.shape == (0, CONFIG.num_mel_bins)
    decoded, num_samples = dmel_from_bytes(dmel_to_bytes(values, 0))
    assert decoded.shape == values.shape
    assert num_samples == 0
