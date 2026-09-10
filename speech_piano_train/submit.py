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
    if execution.container_runtime != "pyxis":
        image = execution.container_path
        if not image.is_file():
            raise FileNotFoundError(image)
    if not (prepared / "metadata.json").is_file():
        raise FileNotFoundError(prepared / "metadata.json")
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
    inner = dedent(
        f"""\
        set -euo pipefail
        cd /workspace/speech-piano-train

        detected=$(python -c 'import torch; print(torch.cuda.device_count())')
        if (( detected != {execution.gpus} )); then
            echo "Expected {execution.gpus} GPUs, found $detected" >&2
            exit 1
        fi

        exec accelerate launch \\
            --config_file config/accelerate-fsdp.yaml \\
            --num_processes "$detected" \\
            "$(command -v speech-piano-train)" \\
            --config-snapshot {shlex.quote(str(snapshot))} \\
            --run-dir {shlex.quote(str(run_dir))}
        """
    ).strip()
    if execution.container_runtime == "pyxis":
        command = [
            f"--container-image={execution.container_reference}",
            f"--container-mounts={prepared}:{prepared}:ro,{run_dir}:{run_dir}",
            "--container-workdir=/workspace/speech-piano-train",
            "--container-mount-home",
            "/bin/bash",
            "-c",
            inner,
        ]
        environment_setup = [
            "set -a",
            f"source {shlex.quote(str(execution.environment_file))}",
            "set +a",
        ]
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
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={experiment}",
            f"#SBATCH --chdir={run_dir}",
            "#SBATCH --output=slurm/%x-%j.out",
            execution.slurm_directives.strip(),
            execution.gpu_directive.strip(),
            "",
            "set -euo pipefail",
            *environment_setup,
            f"exec srun --ntasks=1 {shlex.join(command)}",
            "",
        ]
    )


def validate_directives(value: str) -> None:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines or any(not line.startswith("#SBATCH ") for line in lines):
        raise ValueError("Slurm config must contain only #SBATCH directives")
