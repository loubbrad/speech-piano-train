from __future__ import annotations

import base64
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder


@dataclass(frozen=True)
class CodecConfig:
    sample_rate: int = 48_000
    channels: int = 2
    hop_samples: int = 1_920
    stft_size: int = 960
    stft_hop: int = 480
    latent_size: int = 256
    num_codebooks: int = 12
    codebook_size: int = 1_024
    ratios: tuple[tuple[int, int], ...] = (
        (1, 2),
        (1, 2),
        (1, 3),
        (1, 2),
        (1, 2),
        (2, 2),
        (2, 1),
    )
    multipliers: tuple[int, ...] = (2, 1, 2, 1, 1, 2, 1)
    encode_halo: tuple[int, int] = (14, 2)
    decode_halo: tuple[int, int] = (14, 1)


CONFIG = CodecConfig()
TOKEN_FILE_SCHEMA = 1


def _conv_padding(
    kernel: tuple[int, int], stride: tuple[int, int]
) -> tuple[int, int, int, int]:
    time = kernel[0] - 1
    time_left = max(kernel[0] - stride[0], 0)
    freq = max(kernel[1] - stride[1], 0)
    freq_left = freq // 2
    return freq_left, freq - freq_left, time_left, time - time_left


class _Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self.padding = _conv_padding(kernel, stride)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.padding))


class _ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int],
    ):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel, stride=stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        time_right = self.kernel[0] - self.stride[0]

        total = self.kernel[1] + self.stride[1] - 2
        if self.stride[1] > self.kernel[1] - 1:
            pad_left = self.kernel[1] - 1
        else:
            pad_left = math.ceil(total / 2)
        pad_right = total - pad_left
        freq_left = self.kernel[1] - 1 - pad_left
        freq_right = self.kernel[1] - 1 - pad_right

        time_end = x.shape[2] - time_right if time_right else x.shape[2]
        freq_end = x.shape[3] - freq_right if freq_right else x.shape[3]
        return x[:, :, :time_end, freq_left:freq_end]


def _pool(x: torch.Tensor, stride: tuple[int, int]) -> torch.Tensor:
    if stride == (1, 1):
        return x
    padding = _conv_padding(stride, stride)
    return F.avg_pool2d(
        F.pad(x, padding), stride, stride=stride, count_include_pad=True
    )


class _EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
    ):
        super().__init__()
        kernel = tuple(max(3, 2 * value) for value in stride)
        self.stride = stride
        self.first = _Conv2d(in_channels, in_channels, (3, 3))
        self.second = _Conv2d(in_channels, out_channels, kernel, stride)
        self.shortcut = (
            _Conv2d(in_channels, out_channels, (1, 1))
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.first(F.elu(x))
        y = self.second(F.elu(y))
        shortcut = _pool(x, self.stride)
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        return y + shortcut


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
    ):
        super().__init__()
        kernel = tuple(max(3, 2 * value) for value in stride)
        self.stride = stride
        self.first: _Conv2d | _ConvTranspose2d
        if stride == (1, 1):
            self.first = _Conv2d(in_channels, out_channels, (3, 3))
        else:
            self.first = _ConvTranspose2d(
                in_channels, out_channels, kernel, stride
            )
        self.second = _Conv2d(out_channels, out_channels, (3, 3))
        self.shortcut = (
            _Conv2d(in_channels, out_channels, (1, 1))
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.first(F.elu(x))
        y = self.second(F.elu(y))
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        shortcut = shortcut.repeat_interleave(self.stride[0], 2)
        shortcut = shortcut.repeat_interleave(self.stride[1], 3)
        return y + shortcut


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 32
        self.first = _Conv2d(2, channels, (7, 7))
        blocks = []
        for stride, multiplier in zip(
            CONFIG.ratios[:6], CONFIG.multipliers[:6], strict=True
        ):
            out_channels = channels * multiplier
            blocks.append(_EncoderBlock(channels, out_channels, stride))
            channels = out_channels
        self.channel_blocks = nn.ModuleList(blocks)

        self.joint_block = _EncoderBlock(2 * channels, 256, CONFIG.ratios[6])
        self.bottleneck = _EncoderBlock(256, 256, (1, 1))
        self.output = _Conv2d(1_280, 256, (1, 1))
        self.output_hidden = _Conv2d(1_280, 1_280, (1, 1))
        self.output_final = _Conv2d(1_280, 256, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, time, freq = x.shape
        x = x.reshape(batch * 2, 2, time, freq)
        x = self.first(x)
        for block in self.channel_blocks:
            x = block(x)
        x = x.reshape(batch, 2 * x.shape[1], x.shape[2], x.shape[3])
        x = self.bottleneck(self.joint_block(x))

        x = x.permute(0, 2, 3, 1).flatten(2).transpose(1, 2).unsqueeze(3)
        direct = self.output(x)
        residual = self.output_hidden(F.elu(x))
        residual = self.output_final(F.elu(residual))
        return (direct + residual).squeeze(3)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = _Conv2d(256, 2_560, (1, 1))
        self.input_hidden = _Conv2d(256, 2_560, (1, 1))
        self.input_final = _Conv2d(2_560, 2_560, (1, 1))
        self.bottleneck = _DecoderBlock(512, 512, (1, 1))
        self.joint_block = _DecoderBlock(512, 1_024, (2, 1))

        channels = 512
        blocks = []
        for stride, multiplier in zip(
            reversed(CONFIG.ratios[:6]),
            reversed(CONFIG.multipliers[:6]),
            strict=True,
        ):
            out_channels = channels // multiplier
            blocks.append(_DecoderBlock(channels, out_channels, stride))
            channels = out_channels
        self.channel_blocks = nn.ModuleList(blocks)
        self.output = _Conv2d(channels, 2, (7, 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(3)
        direct = self.input(x)
        residual = self.input_hidden(x)
        x = direct + self.input_final(F.elu(residual))

        batch, _, time, _ = x.shape
        x = x.squeeze(3).transpose(1, 2).reshape(batch, time, 5, 512)
        x = x.permute(0, 3, 1, 2)
        x = self.joint_block(self.bottleneck(x))

        x = x.reshape(batch * 2, x.shape[1] // 2, x.shape[2], x.shape[3])
        for block in self.channel_blocks:
            x = block(x)
        x = self.output(F.elu(x))
        x = x.reshape(batch, 4, x.shape[2], x.shape[3])
        return x[:, :, 4:]


class _RVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "embedding",
            torch.empty(
                CONFIG.num_codebooks,
                CONFIG.codebook_size,
                CONFIG.latent_size,
            ),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        residual = x
        codes = []
        for embedding in self.embedding:
            distance = (
                residual.square().sum(-1, keepdim=True)
                - 2 * residual @ embedding.T
                + embedding.square().sum(-1)
            )
            code = distance.argmin(-1)
            residual = residual - F.embedding(code, embedding)
            codes.append(code)
        return torch.stack(codes, 1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        values = [
            F.embedding(codes[:, index], embedding)
            for index, embedding in enumerate(self.embedding)
        ]
        return torch.stack(values).sum(0).transpose(1, 2)


class SpectroStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.quantizer = _RVQ()
        self.decoder = _Decoder()
        self.register_buffer(
            "stft_window",
            torch.hann_window(CONFIG.stft_size, periodic=True),
            persistent=False,
        )
        window = self.stft_window
        denominator = window.square().reshape(2, CONFIG.stft_hop).sum(0)
        inverse = torch.where(
            denominator.repeat(2) == 0,
            0,
            window / denominator.repeat(2),
        )
        self.register_buffer("istft_window", inverse, persistent=False)

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        audio = F.pad(audio.float(), (0, CONFIG.stft_size - 1))
        frames = audio.unfold(2, CONFIG.stft_size, CONFIG.stft_hop)
        spectrum = torch.fft.rfft(frames * self.stft_window, dim=-1)[..., :-1]
        spectrum = spectrum.permute(0, 2, 3, 1).contiguous()
        values = torch.view_as_real(spectrum).flatten(3)
        return values.permute(0, 3, 1, 2)

    def _istft(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[2] == 0:
            return features.new_empty((features.shape[0], 2, 0))
        values = features.permute(0, 2, 3, 1).contiguous()
        spectrum = torch.view_as_complex(
            values.reshape(*values.shape[:3], 2, 2)
        )
        spectrum = F.pad(spectrum, (0, 0, 0, 1))
        frames = torch.fft.irfft(spectrum, n=CONFIG.stft_size, dim=2)
        frames = frames * self.istft_window[None, None, :, None]
        frames = frames.permute(0, 3, 2, 1).flatten(1, 2)
        width = (spectrum.shape[1] - 1) * CONFIG.stft_hop + CONFIG.stft_size
        audio = F.fold(
            frames,
            output_size=(1, width),
            kernel_size=(1, CONFIG.stft_size),
            stride=(1, CONFIG.stft_hop),
        )
        return audio[:, :, 0, : -CONFIG.stft_hop]

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        _check_audio(audio)
        return self.quantizer.encode(self.encoder(self._stft(audio)))

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        _check_codes(codes)
        embeddings = self.quantizer.decode(codes)
        return self._istft(self.decoder(embeddings))


def load_codec(
    weights_dir: str | Path, device: str | torch.device = "cpu"
) -> SpectroStream:
    weights_dir = Path(weights_dir)
    codec = SpectroStream()
    state = load_file(weights_dir / "encoder.safetensors")
    state.update(load_file(weights_dir / "decoder.safetensors"))
    codec.load_state_dict(state)
    return codec.requires_grad_(False).to(device).eval()


def _check_audio(audio: torch.Tensor) -> None:
    if audio.ndim != 3 or audio.shape[1] != CONFIG.channels:
        raise ValueError(
            f"Expected audio shape [B, 2, samples], got {audio.shape}"
        )
    if not audio.is_floating_point():
        raise TypeError("Audio must be floating point")


def _check_codes(codes: torch.Tensor) -> None:
    if codes.ndim != 3 or codes.shape[1] != CONFIG.num_codebooks:
        raise ValueError(
            f"Expected codes shape [B, 12, frames], got {codes.shape}"
        )
    if codes.is_floating_point():
        raise TypeError("Codec tokens must be integers")
    if codes.numel() and (
        codes.min() < 0 or codes.max() >= CONFIG.codebook_size
    ):
        raise ValueError("Codec tokens must be in [0, 1023]")


def _stereo(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or audio.shape[0] not in (1, 2):
        raise ValueError(
            f"Expected mono or stereo audio [channels, samples], got {audio.shape}"
        )
    return audio.repeat(2, 1) if audio.shape[0] == 1 else audio


def _run_buckets(
    values: list[torch.Tensor], batch_size: int, function
) -> list[torch.Tensor]:
    output: list[torch.Tensor | None] = [None] * len(values)
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        buckets[value.shape[-1]].append(index)
    for indices in buckets.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = torch.stack([values[index] for index in selected])
            result = function(batch)
            for index, item in zip(selected, result, strict=True):
                output[index] = item.cpu()
    return [item for item in output if item is not None]


@torch.inference_mode()
def encode_wav(
    codec: SpectroStream,
    audio: torch.Tensor,
    core_frames: int = 250,
    batch_size: int = 8,
) -> torch.Tensor:
    """Encode one full waveform as haloed, batched windows."""
    if core_frames < 1 or batch_size < 1:
        raise ValueError("core_frames and batch_size must be positive")
    audio = _stereo(audio)
    num_samples = audio.shape[-1]
    content_frames = math.ceil(num_samples / CONFIG.hop_samples)
    # Decoding T tokens produces T - 1 audio frames because of lookahead.
    total_frames = content_frames + 1
    audio = F.pad(audio, (0, total_frames * CONFIG.hop_samples - num_samples))

    left, right = CONFIG.encode_halo
    windows = []
    crops = []
    for core_start in range(0, total_frames, core_frames):
        core_end = min(total_frames, core_start + core_frames)
        context_start = max(0, core_start - left)
        context_end = min(total_frames, core_end + right)
        windows.append(
            audio[
                :,
                context_start * CONFIG.hop_samples : context_end
                * CONFIG.hop_samples,
            ]
        )
        local_start = core_start - context_start
        crops.append((local_start, local_start + core_end - core_start))

    device = next(codec.parameters()).device
    encoded = _run_buckets(
        windows, batch_size, lambda value: codec.encode(value.to(device))
    )
    return torch.cat(
        [
            value[:, start:end]
            for value, (start, end) in zip(encoded, crops, strict=True)
        ],
        dim=1,
    ).long()


@torch.inference_mode()
def decode_wav(
    codec: SpectroStream,
    codes: torch.Tensor,
    core_frames: int = 250,
    batch_size: int = 8,
    num_samples: int | None = None,
) -> torch.Tensor:
    """Decode one full token stream as haloed, batched windows."""
    if codes.ndim != 2:
        raise ValueError(
            f"Expected codec tokens [12, frames], got {codes.shape}"
        )
    _check_codes(codes.unsqueeze(0))
    if core_frames < 1 or batch_size < 1:
        raise ValueError("core_frames and batch_size must be positive")
    output_frames = max(0, codes.shape[1] - 1)
    if output_frames == 0:
        return torch.empty((CONFIG.channels, 0), dtype=torch.float32)

    left, right = CONFIG.decode_halo
    windows = []
    crops = []
    for core_start in range(0, output_frames, core_frames):
        core_end = min(output_frames, core_start + core_frames)
        context_start = max(0, core_start - left)
        context_end = min(codes.shape[1], core_end + right)
        windows.append(codes[:, context_start:context_end])
        local_start = (core_start - context_start) * CONFIG.hop_samples
        crops.append(
            (
                local_start,
                local_start + (core_end - core_start) * CONFIG.hop_samples,
            )
        )

    device = next(codec.parameters()).device
    decoded = _run_buckets(
        windows, batch_size, lambda value: codec.decode(value.to(device))
    )
    audio = torch.cat(
        [
            value[:, start:end]
            for value, (start, end) in zip(decoded, crops, strict=True)
        ],
        dim=1,
    ).float()
    if num_samples is not None:
        if num_samples < 0 or num_samples > audio.shape[1]:
            raise ValueError(f"Invalid num_samples {num_samples}")
        audio = audio[:, :num_samples]
    return audio


def codec_tokens_to_bytes(codes: torch.Tensor, num_samples: int) -> bytes:
    """Serialize frame-major tokens, packing each four 10-bit codes in 5 bytes."""
    if codes.ndim != 2:
        raise ValueError(
            f"Expected codec tokens [12, frames], got {codes.shape}"
        )
    _check_codes(codes.unsqueeze(0))
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    values = codes.T.contiguous().cpu().numpy().astype(np.uint64, copy=False)
    values = values.reshape(-1, 3, 4)
    packed = (
        values[..., 0]
        | values[..., 1] << 10
        | values[..., 2] << 20
        | values[..., 3] << 30
    )
    shifts = np.arange(0, 40, 8, dtype=np.uint64)
    raw = ((packed[..., None] >> shifts) & 0xFF).astype(np.uint8).tobytes()
    return json.dumps(
        {
            "schema_version": TOKEN_FILE_SCHEMA,
            "num_samples": num_samples,
            "codes_b64": base64.b64encode(raw).decode("ascii"),
        },
        separators=(",", ":"),
    ).encode()


def codec_tokens_from_bytes(payload: bytes) -> tuple[torch.Tensor, int]:
    """Deserialize the schema-v1 codec-token format."""
    value = json.loads(payload)
    if value.get("schema_version") != TOKEN_FILE_SCHEMA:
        raise ValueError("Unsupported codec-token schema")
    num_samples = value.get("num_samples")
    if not isinstance(num_samples, int) or num_samples < 0:
        raise ValueError("Invalid num_samples")
    raw = base64.b64decode(value["codes_b64"], validate=True)
    frame_bytes = CONFIG.num_codebooks * 10 // 8
    if len(raw) % frame_bytes:
        raise ValueError("Invalid codec-token payload length")
    groups = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3, 5)
    shifts = np.arange(0, 40, 8, dtype=np.uint64)
    packed = (groups.astype(np.uint64) << shifts).sum(-1)
    values = np.stack(
        [(packed >> shift) & 0x3FF for shift in range(0, 40, 10)], axis=-1
    ).reshape(-1, CONFIG.num_codebooks)
    codes = torch.from_numpy(values.copy()).T.contiguous().long()
    _check_codes(codes.unsqueeze(0))
    return codes, num_samples


def save_codec_tokens(
    path: str | Path, codes: torch.Tensor, num_samples: int
) -> None:
    Path(path).write_bytes(codec_tokens_to_bytes(codes, num_samples))


def load_codec_tokens(path: str | Path) -> tuple[torch.Tensor, int]:
    return codec_tokens_from_bytes(Path(path).read_bytes())


def encode_file(
    codec: SpectroStream,
    audio_path: str | Path,
    token_path: str | Path,
    core_frames: int = 250,
    batch_size: int = 8,
    resample: bool = False,
) -> None:
    decoder = AudioDecoder(
        audio_path,
        sample_rate=CONFIG.sample_rate if resample else None,
    )
    source_sample_rate = decoder.metadata.sample_rate
    if not resample and source_sample_rate != CONFIG.sample_rate:
        raise ValueError(
            f"Expected {CONFIG.sample_rate} Hz audio, got {source_sample_rate}; "
            "enable resampling to convert it"
        )
    audio = decoder.get_all_samples().data
    codes = encode_wav(codec, audio, core_frames, batch_size)
    save_codec_tokens(token_path, codes, audio.shape[1])


def decode_file(
    codec: SpectroStream,
    token_path: str | Path,
    audio_path: str | Path,
    core_frames: int = 250,
    batch_size: int = 8,
) -> None:
    codes, num_samples = load_codec_tokens(token_path)
    audio = decode_wav(codec, codes, core_frames, batch_size, num_samples)
    AudioEncoder(audio, sample_rate=CONFIG.sample_rate).to_file(str(audio_path))
