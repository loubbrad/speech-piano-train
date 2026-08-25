#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

readarray -t execution_config < <(
    uv run --no-sync python <<'PY'
from speech_piano_train.config import load_config

execution = load_config().execution
print(execution.container_runtime)
print(execution.container_path)
PY
)
container_runtime="${execution_config[0]}"
container_image="${execution_config[1]}"
container_dir="$(dirname -- "$container_image")"
deps_image="$container_dir/speech-piano-deps.sif"
mkdir -p "$container_dir"

if [[ ${1-} == --deps ]]; then
    shift
    "$container_runtime" build "$@" \
        "$deps_image" \
        containers/speech-piano-deps.def
fi

if [[ ! -f "$deps_image" ]]; then
    echo "Missing $deps_image; rebuild with --deps" >&2
    exit 1
fi

exec "$container_runtime" build "$@" \
    --build-arg "deps_image=$deps_image" \
    "$container_image" \
    containers/speech-piano.def
