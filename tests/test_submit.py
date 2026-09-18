import subprocess
from pathlib import Path

from speech_piano_train.config import load_config
from speech_piano_train.submit import (
    batch_script,
    prepare_submission,
    preview_submission,
)


def configured(
    tmp_path: Path,
    *,
    runtime: str = "singularity",
    representation: str = "midi_text",
):
    dataset = tmp_path / "dataset"
    prepared = tmp_path / f"prepared/interleaved/{representation}"
    experiments = tmp_path / "experiments"
    image = tmp_path / "speech-piano.sif"
    environment = tmp_path / ".env"
    dataset.mkdir()
    prepared.mkdir(parents=True)
    image.touch()
    environment.touch()
    (prepared / "metadata.json").write_text("{}", encoding="utf-8")
    local = tmp_path / "local.yaml"
    local.write_text(
        f"""\
data:
  dataset_dir: {dataset}
  prepared_dir: {tmp_path / "prepared"}
  representation: {representation}
execution:
  experiments_dir: {experiments}
  container_image: {image}
  environment_file: {environment}
  container_runtime: {runtime}
  slurm: |
    #SBATCH --partition=gpu
""",
        encoding="utf-8",
    )
    return load_config(local_path=local)


def test_batch_script_is_valid_bash(tmp_path: Path) -> None:
    config = configured(tmp_path)
    run_dir = tmp_path / "experiments/run"
    run_dir.mkdir(parents=True)
    script = batch_script("run", run_dir, config)

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "accelerate launch" in script
    assert "singularity exec" in script
    assert "--exact" not in script
    assert "config/accelerate-fsdp.yaml" in script
    assert '--num_processes "$detected"' in script
    assert '"Pianoteq 8 STAGE" --headless --activate' not in script
    assert "unset PIANOTEQ_KEY" in script
    assert str(config.data.prepared_path) in script
    assert "speech-piano-train" in script


def test_audio_batch_script_activates_pianoteq(tmp_path: Path) -> None:
    config = configured(tmp_path, representation="mel")
    run_dir = tmp_path / "experiments/run"
    run_dir.mkdir(parents=True)

    script = batch_script("run", run_dir, config)

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert '"Pianoteq 8 STAGE" --headless --activate "$PIANOTEQ_KEY"' in script
    assert script.index("--activate") < script.index("unset PIANOTEQ_KEY")


def test_batch_script_uses_pyxis_for_enroot(tmp_path: Path) -> None:
    config = configured(tmp_path, runtime="pyxis")
    run_dir = tmp_path / "experiments/run"
    run_dir.mkdir(parents=True)
    script = batch_script("run", run_dir, config)

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "--container-image=" in script
    assert "--container-mounts=" in script
    assert "--container-workdir=/workspace/speech-piano-train" in script
    assert "--no-container-mount-home" in script
    assert f"{config.execution.environment_file}:/workspace/speech-piano-train/.env:ro" in script
    assert "source /workspace/speech-piano-train/.env" in script
    assert "--main_process_port" in script
    assert "singularity exec" not in script
    assert "srun" not in script


def test_pyxis_mounts_local_model_read_only(tmp_path: Path) -> None:
    config = configured(tmp_path, runtime="pyxis")
    model = tmp_path / "model"
    model.mkdir()
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"name": str(model)})}
    )
    run_dir = tmp_path / "experiments/run"
    run_dir.mkdir(parents=True)

    script = batch_script("run", run_dir, config)

    assert f"{model}:{model}:ro" in script


def test_prepare_submission_snapshots_config(tmp_path: Path) -> None:
    config = configured(tmp_path)

    run_dir = prepare_submission("interleaved", config)

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "job.sh").is_file()
    assert (run_dir / "slurm").is_dir()


def test_preview_submission_does_not_create_run_directory(tmp_path: Path) -> None:
    config = configured(tmp_path)

    script = preview_submission("interleaved", config)

    assert "#SBATCH --job-name=interleaved" in script
    assert not (tmp_path / "experiments/interleaved").exists()


def test_prepare_submission_requires_local_model(tmp_path: Path) -> None:
    config = configured(tmp_path, runtime="pyxis")
    missing = tmp_path / "missing-model"
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"name": str(missing)})}
    )

    try:
        prepare_submission("interleaved", config)
    except FileNotFoundError as error:
        assert error.args == (missing,)
    else:
        raise AssertionError("missing local model was accepted")
