from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from pathlib import Path
from textwrap import dedent

from speech_piano_train.config import AppConfig, load_config, write_config

EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one training run to Slurm.")
    parser.add_argument("experiment")
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not EXPERIMENT_RE.fullmatch(args.experiment):
        parser.error("experiment must contain only letters, digits, '.', '_', and '-'")

    config = load_config(args.config_file)
    run_dir = prepare_submission(args.experiment, config)
    if args.dry_run:
        print(run_dir / "job.sh")
        return
    subprocess.run(["sbatch", str(run_dir / "job.sh")], check=True)


def prepare_submission(experiment: str, config: AppConfig) -> Path:
    execution = config.execution
    prepared = config.data.prepared_path
    environment = execution.environment_file
    if execution.container_runtime == "pyxis":
        image_value = Path(execution.container_image or "")
        if image_value.is_absolute() and not image_value.is_file():
            raise FileNotFoundError(image_value)
    else:
        image = execution.container_path
        if not image.is_file():
            raise FileNotFoundError(image)
    if not (prepared / "metadata.json").is_file():
        raise FileNotFoundError(prepared / "metadata.json")
    model_path = Path(config.model.name)
    if model_path.is_absolute() and not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not environment.is_file():
        raise FileNotFoundError(environment)
    validate_directives(execution.slurm_directives)
    validate_directives(execution.gpu_directive)

    run_dir = execution.experiments_path / experiment
    run_dir.mkdir(parents=True)
    (run_dir / "slurm").mkdir()
    write_config(run_dir / "config.yaml", config)
    job = run_dir / "job.sh"
    job.write_text(batch_script(experiment, run_dir, config), encoding="utf-8")
    job.chmod(0o755)
    return run_dir


def batch_script(experiment: str, run_dir: Path, config: AppConfig) -> str:
    execution = config.execution
    prepared = config.data.prepared_path
    snapshot = run_dir / "config.yaml"
    # Pin the container to the first `gpus` devices. Pyxis exposes every GPU on
    # the node regardless of --gres, so without this the GPU-count guard sees
    # the whole node. For a full-node request this is a no-op (0..N-1 = all).
    gpu_list = ",".join(str(index) for index in range(execution.gpus))
    container_directives: list[str] = []
    inner = dedent(
        f"""\
        set -euo pipefail
        export HOME=/tmp/speech-piano-home
        mkdir -p "$HOME"
        cd /workspace/speech-piano-train

        detected=$(python -c 'import torch; print(torch.cuda.device_count())')
        if (( detected != {execution.gpus} )); then
            echo "Expected {execution.gpus} GPUs, found $detected" >&2
            exit 1
        fi

        exec accelerate launch \\
            --config_file config/accelerate-fsdp.yaml \\
            --num_processes "$detected" \\
            --main_process_port "$MASTER_PORT" \\
            "$(command -v speech-piano-train)" \\
            --config-snapshot {shlex.quote(str(snapshot))} \\
            --run-dir {shlex.quote(str(run_dir))}
        """
    ).strip()
    if execution.container_runtime == "pyxis":
        mounts = [
            f"{prepared}:{prepared}:ro",
            f"{run_dir}:{run_dir}",
            f"{execution.environment_file}:/workspace/speech-piano-train/.env:ro",
        ]
        model_path = Path(config.model.name)
        if model_path.is_absolute():
            mounts.append(f"{model_path}:{model_path}:ro")
        image = execution.container_reference.replace("#", r"\#")
        container_directives = [
            f"#SBATCH --container-image={image}",
            f"#SBATCH --container-mounts={','.join(mounts)}",
            "#SBATCH --container-workdir=/workspace/speech-piano-train",
            "#SBATCH --no-container-mount-home",
        ]
        environment_setup = [
            "set -a",
            "source /workspace/speech-piano-train/.env",
            "set +a",
        ]
        launch = inner
    else:
        command = [
            execution.container_runtime,
            "exec",
            "--nv",
            "--env-file",
            str(execution.environment_file),
            "--bind",
            f"{prepared}:{prepared}:ro",
            "--bind",
            f"{run_dir}:{run_dir}",
            execution.container_reference,
            "/bin/bash",
            "-c",
            inner,
        ]
        environment_setup = []
        launch = f"exec {shlex.join(command)}"
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={experiment}",
            f"#SBATCH --chdir={run_dir}",
            "#SBATCH --output=slurm/%x-%j.out",
            execution.slurm_directives.strip(),
            execution.gpu_directive.strip(),
            *container_directives,
            "",
            "set -euo pipefail",
            *environment_setup,
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export TOKENIZERS_PARALLELISM=false",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            f"export CUDA_VISIBLE_DEVICES={gpu_list}",
            'export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"',
            launch,
            "",
        ]
    )


def validate_directives(value: str) -> None:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines or any(not line.startswith("#SBATCH ") for line in lines):
        raise ValueError("Slurm config must contain only #SBATCH directives")
