#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=common.sh
source "$repo_dir/scripts/irmak/common.sh"
cd "$repo_dir"

environment_file="$repo_dir/.env"
if [[ ! -f $environment_file ]]; then
    echo "Missing $environment_file; create it with WANDB_API_KEY=..." >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "$environment_file"
set +a
: "${WANDB_API_KEY:?Set WANDB_API_KEY in $environment_file}"
if [[ ! -f $IRMAK_CONTAINER ]]; then
    echo "Missing $IRMAK_CONTAINER; run scripts/irmak/refresh-container.sh first" >&2
    exit 1
fi

case ${1-interleaved} in
    interleaved|separated) conditions=("$1") ;;
    --dry-run) conditions=(interleaved); dry_run=1 ;;
    *)
        echo "Usage: $0 [interleaved|separated|--dry-run]" >&2
        exit 2
        ;;
esac

submit_env="$repo_dir/.irmak-submit-env"
if [[ ! -x $submit_env/bin/python ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is required to create the small submission environment" >&2
        exit 1
    fi
    echo "Creating the submission environment at $submit_env"
    conda create --yes --prefix "$submit_env" \
        python=3.12 pip setuptools wheel 'pydantic>=2' pyyaml
fi

"$submit_env/bin/python" -m pip install \
    --quiet --no-build-isolation --no-deps --editable "$repo_dir"

for condition in "${conditions[@]}"; do
    command=(
        "$submit_env/bin/speech-piano-submit"
        "qwen35-9b-$condition"
        --config-file "$repo_dir/config/$condition.yaml"
    )
    if [[ ${dry_run:-0} == 1 ]]; then
        command+=(--dry-run)
    fi
    "${command[@]}"
done
