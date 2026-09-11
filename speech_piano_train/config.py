from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def resolve_paths(cls, value: Any) -> Any:
        if isinstance(value, Path):
            path = value if value.is_absolute() else REPOSITORY_ROOT / value
            return path.resolve()
        return value


class DataConfig(ConfigModel):
    dataset_dir: Path | None
    prepared_dir: Path | None
    variant: Literal["interleaved", "separated"]
    seed: int
    sequence_length: int = Field(gt=0)
    workers: int = Field(gt=0)

    @property
    def dataset_path(self) -> Path:
        assert self.dataset_dir is not None
        return self.dataset_dir

    @property
    def prepared_path(self) -> Path:
        assert self.prepared_dir is not None
        return self.prepared_dir / self.variant


class ModelConfig(ConfigModel):
    name: str


class TrainConfig(ConfigModel):
    seed: int
    micro_batch_size: int = Field(gt=0)
    tokens_per_update: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    betas: tuple[float, float]
    eps: float = Field(gt=0)
    warmup_ratio: float = Field(ge=0, lt=1)
    max_grad_norm: float = Field(gt=0)
    checkpoint_interval: int = Field(gt=0)
    num_workers: int = Field(ge=0)


class WandbConfig(ConfigModel):
    project: str
    name: str | None
    mode: Literal["online", "offline", "disabled"]


class ExecutionConfig(ConfigModel):
    experiments_dir: Path | None
    container_image: str | None
    environment_file: Path
    container_runtime: Literal["apptainer", "singularity", "pyxis"]
    gpus: int = Field(gt=0)
    gpu_directive: str
    slurm: str | None

    @property
    def experiments_path(self) -> Path:
        assert self.experiments_dir is not None
        return self.experiments_dir

    @property
    def container_path(self) -> Path:
        assert self.container_image is not None
        path = Path(self.container_image)
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        return path.resolve()

    @property
    def container_reference(self) -> str:
        assert self.container_image is not None
        if self.container_runtime == "pyxis":
            path = Path(self.container_image)
            if path.is_absolute():
                return str(path.resolve())
            return self.container_image
        return str(self.container_path)

    @property
    def slurm_directives(self) -> str:
        assert self.slurm is not None
        return self.slurm


class AppConfig(ConfigModel):
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    wandb: WandbConfig
    execution: ExecutionConfig


def load_config(
    override_path: str | Path | None = None,
    *,
    base_path: str | Path | None = None,
    local_path: str | Path | None = None,
) -> AppConfig:
    base = Path(base_path) if base_path else REPOSITORY_ROOT / "config/base.yaml"
    local = Path(local_path) if local_path else REPOSITORY_ROOT / "config/local.yaml"
    value = _read_yaml(base)
    if local.is_file():
        value = _merge(value, _read_yaml(local))
    if override_path is not None:
        value = _merge(value, _read_yaml(Path(override_path)))
    return AppConfig.model_validate(value)


def load_config_file(path: str | Path) -> AppConfig:
    return AppConfig.model_validate(_read_yaml(Path(path)))


def write_config(path: str | Path, config: AppConfig) -> None:
    Path(path).write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config must contain a mapping: {path}")
    return value


def _merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
