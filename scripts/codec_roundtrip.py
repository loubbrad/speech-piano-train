"""Encode a WAV to codec tokens, reload them, and decode them to a WAV."""

import argparse
from pathlib import Path

import torch

from speech_piano_train.codec import decode_file, encode_file, load_codec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input 48 kHz mono or stereo WAV")
    parser.add_argument("weights", type=Path, help="SpectroStream weights directory")
    parser.add_argument("--tokens", type=Path, help="Output codec-token JSON")
    parser.add_argument("--output", type=Path, help="Output decoded WAV")
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample input audio to 48 kHz when needed",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda when available)",
    )
    args = parser.parse_args()

    stem = args.audio.with_suffix("")
    tokens = args.tokens or stem.with_name(stem.name + ".codec.json")
    output = args.output or stem.with_name(stem.name + ".decoded.wav")

    codec = load_codec(args.weights, args.device)
    encode_file(codec, args.audio, tokens, resample=args.resample)
    decode_file(codec, tokens, output)

    print(f"Codec tokens: {tokens}")
    print(f"Decoded WAV:  {output}")


if __name__ == "__main__":
    main()
