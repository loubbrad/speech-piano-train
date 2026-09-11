#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=common.sh
source "$repo_dir/scripts/irmak/common.sh"
cd "$repo_dir"

revision=$(git rev-parse HEAD)
export IRMAK_IMAGE_REVISION=$revision
registry=${IRMAK_IMAGE_REPOSITORY%%/*}
image_name=${IRMAK_IMAGE_REPOSITORY#*/}
image_uri="docker://${registry}#${image_name}:${revision}"
revision_image="$IRMAK_CONTAINER_DIR/speech-piano-${revision}.sqsh"

mkdir -p "$IRMAK_CONTAINER_DIR"
irmak_srun_args

if [[ ! -f $revision_image ]]; then
    partial="$revision_image.partial"
    if [[ -e $partial ]]; then
        echo "Remove the incomplete image before retrying: $partial" >&2
        exit 1
    fi
    echo "Importing $image_uri"
    srun "${IRMAK_SRUN_ARGS[@]}" \
        --gres=gpu:1 \
        --cpus-per-task=16 \
        --time=01:00:00 \
        enroot import --output "$partial" "$image_uri"
    mv -- "$partial" "$revision_image"
else
    echo "Using existing image for commit $revision"
fi

echo "Checking the image on eight GPUs"
srun "${IRMAK_SRUN_ARGS[@]}" \
    --gres=gpu:8 \
    --cpus-per-task=32 \
    --time=00:15:00 \
    --container-image="$revision_image" \
    --no-container-mount-home \
    --container-env=IRMAK_IMAGE_REVISION \
    bash -c '
        set -euo pipefail
        actual=$(cat /opt/speech-piano/image-revision)
        [[ $actual == "$IRMAK_IMAGE_REVISION" ]] || {
            echo "Image revision mismatch: expected $IRMAK_IMAGE_REVISION, found $actual" >&2
            exit 1
        }
        python - <<"PY"
import causal_conv1d
import fla
import torch
from transformers import Qwen3_5ForCausalLM

count = torch.cuda.device_count()
assert count == 8, f"expected 8 GPUs, found {count}"
print(f"torch={torch.__version__} cuda={torch.version.cuda} GPUs={count}")
for index in range(count):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    assert "H100" in name, f"GPU {index} is not an H100: {name}"
    assert capability == (9, 0), f"unexpected capability for GPU {index}: {capability}"
    print(index, name, capability)
PY
        hf version
        nvidia-smi topo -m
    '

ln -sfn -- "$(basename -- "$revision_image")" "$IRMAK_CONTAINER"
echo "Container ready: $revision_image"
