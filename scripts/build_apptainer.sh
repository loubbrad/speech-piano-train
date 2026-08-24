#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

container_image="$(
    python <<'PY'
from speech_piano_train.config import load_config

print(load_config().execution.container_path)
PY
)"
container_dir="$(dirname -- "$container_image")"
deps_image="$container_dir/speech-piano-deps.sif"
mkdir -p "$container_dir"

if [[ ${1-} == --deps ]]; then
    shift
    apptainer build "$@" \
        "$deps_image" \
        containers/speech-piano-deps.def
fi

if [[ ! -f "$deps_image" ]]; then
    echo "Missing $deps_image; rebuild with --deps" >&2
    exit 1
fi

exec apptainer build "$@" \
    --build-arg "deps_image=$deps_image" \
    "$container_image" \
    containers/speech-piano.def
