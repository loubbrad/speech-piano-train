from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from speech_piano_train.config import load_config


def test_loads_local_paths_and_experiment_override(tmp_path: Path) -> None:
    local = tmp_path / "local.yaml"
    local.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset_dir": str(tmp_path / "dataset"),
                    "prepared_dir": str(tmp_path / "prepared"),
                },
                "execution": {
                    "experiments_dir": str(tmp_path / "experiments"),
                    "container_image": str(tmp_path / "image.sif"),
                    "slurm": "#SBATCH --partition=gpu\n",
                },
            }
        ),
        encoding="utf-8",
    )

    experiment = tmp_path / "separated.yaml"
    experiment.write_text("data:\n  variant: separated\n", encoding="utf-8")
    config = load_config(experiment, local_path=local)

    assert config.data.variant == "separated"
    assert config.data.prepared_path == (tmp_path / "prepared/separated/midi_text")
    assert config.execution.gpus == 4
    assert config.execution.container_runtime == "singularity"
    assert config.execution.container_reference == str(tmp_path / "image.sif")
    assert config.train.micro_batch_size == 1
    assert config.model.name == "Qwen/Qwen3.5-9B-Base"


def test_rejects_unknown_config_keys(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text("train:\n  typo: true\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(override, local_path=tmp_path / "missing.yaml")


def test_rejects_mel_for_separated_documents(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text(
        "data:\n  variant: separated\n  representation: mel\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="only supports interleaved"):
        load_config(override, local_path=tmp_path / "missing.yaml")


def test_pyxis_keeps_registry_image_reference(tmp_path: Path) -> None:
    local = tmp_path / "local.yaml"
    local.write_text(
        yaml.safe_dump(
            {
                "execution": {
                    "container_runtime": "pyxis",
                    "container_image": "ghcr.io#example/speech-piano:dev",
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(local_path=local)

    assert config.execution.container_reference == (
        "ghcr.io#example/speech-piano:dev"
    )


def test_pyxis_resolves_local_image_symlink(tmp_path: Path) -> None:
    image = tmp_path / "image-sha.sqsh"
    image.touch()
    current = tmp_path / "current.sqsh"
    current.symlink_to(image.name)
    local = tmp_path / "local.yaml"
    local.write_text(
        yaml.safe_dump(
            {
                "execution": {
                    "container_runtime": "pyxis",
                    "container_image": str(current),
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(local_path=local)

    assert config.execution.container_reference == str(image)
