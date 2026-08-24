import subprocess
from pathlib import Path

from speech_piano_train.config import load_config
from speech_piano_train.submit import batch_script, prepare_submission


def configured(tmp_path: Path):
    dataset = tmp_path / "dataset"
    prepared = tmp_path / "prepared/interleaved"
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
execution:
  experiments_dir: {experiments}
  container_image: {image}
  environment_file: {environment}
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
    assert '--num_processes "$detected"' in script
    assert str(config.data.prepared_path) in script
    assert "speech-piano-train" in script


def test_prepare_submission_snapshots_config(tmp_path: Path) -> None:
    config = configured(tmp_path)

    run_dir = prepare_submission("interleaved", config)

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "job.sh").is_file()
    assert (run_dir / "slurm").is_dir()
