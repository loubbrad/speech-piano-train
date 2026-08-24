import argparse
import json
from pathlib import Path

from speech_piano_train.config import load_config, load_config_file
from speech_piano_train.data import prepare_data


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Speech/Piano tokens.")
    parser.add_argument("--config-file", type=Path)
    args = parser.parse_args()
    config = load_config(args.config_file)
    metadata = prepare_data(config.data, config.model)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Continue pretraining Qwen3.5.")
    parser.add_argument("--config-snapshot", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config_file(args.config_snapshot)

    from speech_piano_train.train import train

    train(config, args.run_dir)
