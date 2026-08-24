#!/usr/bin/env bash
#SBATCH --job-name=speech-piano-build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/build-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
mkdir -p slurm
exec ./scripts/build_apptainer.sh "$@" --force

