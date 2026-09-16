#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=common.sh
source "$repo_dir/scripts/irmak/common.sh"

environment_file="$repo_dir/.env"
if [[ ! -f $environment_file ]]; then
    echo "Missing $environment_file; create it with HF_TOKEN=..." >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "$environment_file"
set +a
: "${HF_TOKEN:?Set HF_TOKEN in $environment_file}"

if [[ ! -f $IRMAK_CONTAINER ]]; then
    echo "Missing $IRMAK_CONTAINER; run scripts/irmak/refresh-container.sh first" >&2
    exit 1
fi

mkdir -p \
    "$IRMAK_DOWNLOAD_DIR" \
    "$IRMAK_DATA_ROOT" \
    "$IRMAK_MODEL_ROOT" \
    "$IRMAK_PREPARED_ROOT" \
    "$IRMAK_HF_HOME"

export HF_HOME=$IRMAK_HF_HOME
export HF_HUB_CACHE=$IRMAK_HF_HOME/hub
export IRMAK_DATA_REPOSITORY IRMAK_DATA_ARCHIVE IRMAK_DOWNLOAD_DIR
export IRMAK_DATASET_DIR IRMAK_DATA_ROOT IRMAK_MODEL_REPOSITORY IRMAK_MODEL_DIR
export IRMAK_PREPARED_ROOT

echo "Downloading the corpus and model, then preparing both token streams"
enroot start \
    --env NVIDIA_VISIBLE_DEVICES=void \
    --env HF_TOKEN \
    --env HF_HOME \
    --env HF_HUB_CACHE \
    --env IRMAK_DATA_REPOSITORY \
    --env IRMAK_DATA_ARCHIVE \
    --env IRMAK_DOWNLOAD_DIR \
    --env IRMAK_DATASET_DIR \
    --env IRMAK_DATA_ROOT \
    --env IRMAK_MODEL_REPOSITORY \
    --env IRMAK_MODEL_DIR \
    --env IRMAK_PREPARED_ROOT \
    --mount "$IRMAK_DOWNLOAD_DIR:$IRMAK_DOWNLOAD_DIR:none:x-create=dir,bind" \
    --mount "$IRMAK_DATA_ROOT:$IRMAK_DATA_ROOT:none:x-create=dir,bind" \
    --mount "$IRMAK_MODEL_ROOT:$IRMAK_MODEL_ROOT:none:x-create=dir,bind" \
    --mount "$IRMAK_PREPARED_ROOT:$IRMAK_PREPARED_ROOT:none:x-create=dir,bind" \
    --mount "$IRMAK_HF_HOME:$IRMAK_HF_HOME:none:x-create=dir,bind" \
    --mount "$repo_dir/config:/workspace/speech-piano-train/config:none:bind,ro" \
    "$IRMAK_CONTAINER" bash -s <<'BATCH'
set -euo pipefail

cd /workspace/speech-piano-train

hf auth whoami
hf download "$IRMAK_MODEL_REPOSITORY" \
    --local-dir "$IRMAK_MODEL_DIR"

archive="$IRMAK_DOWNLOAD_DIR/$IRMAK_DATA_ARCHIVE"
if [[ ! -f $IRMAK_DATASET_DIR/manifest.json && \
      ! -f $IRMAK_DATASET_DIR/documents/manifest.json ]]; then
    if [[ -e $IRMAK_DATASET_DIR ]]; then
        echo "Dataset directory exists but has no manifest: $IRMAK_DATASET_DIR" >&2
        exit 1
    fi
    hf download "$IRMAK_DATA_REPOSITORY" "$IRMAK_DATA_ARCHIVE" \
        --type dataset \
        --local-dir "$IRMAK_DOWNLOAD_DIR"
    extract_root="$IRMAK_DATA_ROOT/.speech-piano-extract-login-$$"
    if [[ -e $extract_root ]]; then
        echo "Incomplete extraction already exists: $extract_root" >&2
        exit 1
    fi
    mkdir "$extract_root"
    tar --checkpoint=10000 --checkpoint-action=dot \
        -xzf "$archive" -C "$extract_root"
    echo
    extracted="$extract_root/$(basename "$IRMAK_DATASET_DIR")"
    if [[ ! -f $extracted/manifest.json && \
          ! -f $extracted/documents/manifest.json ]]; then
        echo "Archive did not contain the expected $(basename "$IRMAK_DATASET_DIR") directory" >&2
        exit 1
    fi
    mv "$extracted" "$IRMAK_DATASET_DIR"
    rmdir "$extract_root"
else
    echo "Dataset already extracted: $IRMAK_DATASET_DIR"
fi

for condition in interleaved separated; do
    output="$IRMAK_PREPARED_ROOT/$condition"
    if [[ -f $output/metadata.json ]]; then
        echo "Already prepared: $output"
    elif [[ -e $output ]]; then
        echo "Incomplete prepared directory exists: $output" >&2
        echo "Inspect or move it aside before retrying." >&2
        exit 1
    else
        speech-piano-prepare --config-file "config/$condition.yaml"
    fi
done
BATCH

echo "Data and model are ready under $IRMAK_PROJECT_ROOT"
