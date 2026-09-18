import torch
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)

from speech_piano_train.mel import CONFIG, MelEncoder


def test_mel_shape_and_interior_match_whisper() -> None:
    audio = torch.randn(3_200, generator=torch.Generator().manual_seed(123))
    actual = MelEncoder()(audio)
    expected = torch.from_numpy(
        WhisperFeatureExtractor()._torch_extract_fbank_features(audio.numpy())
    ).T

    assert actual.shape == (20, CONFIG.num_mel_bins)
    assert torch.equal(actual[2:-2], expected[2:-2])


def test_empty_mel() -> None:
    assert MelEncoder()(torch.empty(0)).shape == (0, CONFIG.num_mel_bins)
