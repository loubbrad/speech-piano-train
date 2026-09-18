import torch
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from speech_piano_train.model import AudioQwen3_5ForCausalLM, MelAdapter


def test_audio_model_registers_qwen_checkpoint_conversion() -> None:
    conversions = get_checkpoint_conversion_mapping("AudioQwen3_5ForCausalLM")
    assert conversions is not None
    key = "model.language_model.layers.0.input_layernorm.weight"
    for conversion in conversions:
        key, _ = conversion.rename_source_key(key)
    assert key == "model.layers.0.input_layernorm.weight"


def test_mel_adapter_downsamples_frames_and_backpropagates() -> None:
    adapter = MelAdapter(hidden_size=16, eps=1e-6, initializer_range=0.02)
    values = torch.randn(2, 16, 80)

    output = adapter(values)
    output.sum().backward()

    assert output.shape == (2, 4, 16)
    assert adapter.conv1.weight.grad is not None


def test_mel_adapter_emits_one_position_per_four_frames() -> None:
    adapter = MelAdapter(hidden_size=16, eps=1e-6, initializer_range=0.02)
    output = adapter(torch.zeros((1, 28, 80)))

    assert output.shape == (1, 7, 16)


def test_audio_model_replaces_embeddings_and_masks_audio_loss() -> None:
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"],
        max_position_embeddings=64,
    )
    model = AudioQwen3_5ForCausalLM(config)
    input_ids = torch.randint(0, 32, (2, 8))
    mel_values = torch.zeros((2, 32, 80))
    mel_mask = torch.zeros((2, 8), dtype=torch.bool)
    mel_mask[:, 2:4] = True
    labels = input_ids.clone()
    labels[mel_mask] = -100

    output = model(
        input_ids=input_ids,
        mel_values=mel_values,
        mel_mask=mel_mask,
        labels=labels,
        use_cache=False,
    )
    output.loss.backward()

    assert output.logits.shape == (2, 8, 32)
    assert model.mel_adapter.conv1.weight.grad is not None
