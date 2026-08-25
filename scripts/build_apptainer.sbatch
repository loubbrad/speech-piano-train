#!/usr/bin/env bash
#SBATCH --job-name=speech-piano-build
#SBATCH --account=mt
#SBATCH --partition=sharedp
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/build-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
mkdir -p slurm
exec ./scripts/build_apptainer.sh "$@" --force --fakeroot
