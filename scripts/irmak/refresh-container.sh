#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=common.sh
source "$repo_dir/scripts/irmak/common.sh"
cd "$repo_dir"

registry=${IRMAK_IMAGE_REPOSITORY%%/*}
image_name=${IRMAK_IMAGE_REPOSITORY#*/}
image_uri="docker://${registry}#${image_name}:main"
main_image="$IRMAK_CONTAINER_DIR/speech-piano-main.sqsh"

mkdir -p "$IRMAK_CONTAINER_DIR"
environment_file="$repo_dir/.env"
if [[ ! -f $environment_file ]]; then
    echo "Missing $environment_file; create it with GHCR_TOKEN=..." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$environment_file"
: "${GHCR_TOKEN:?Set GHCR_TOKEN in $environment_file}"
export GHCR_TOKEN
export ENROOT_CONFIG_PATH="$IRMAK_CONTAINER_DIR/enroot-config"
mkdir -p "$ENROOT_CONFIG_PATH"
printf 'machine ghcr.io login %s password $GHCR_TOKEN\n' \
    "${GHCR_USERNAME:-loubbrad}" > "$ENROOT_CONFIG_PATH/.credentials"
chmod 600 "$ENROOT_CONFIG_PATH/.credentials"

irmak_srun_args

partial="$main_image.partial"
if [[ -e $partial ]]; then
    echo "Remove the incomplete image before retrying: $partial" >&2
    exit 1
fi
echo "Importing $image_uri"
srun "${IRMAK_SRUN_ARGS[@]}" \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --time=01:00:00 \
    bash -c '
        set -euo pipefail
        enroot import --output "$1" "$2"
        mv -- "$1" "$3"
    ' bash "$partial" "$image_uri" "$main_image"

ln -sfn -- "$(basename -- "$main_image")" "$IRMAK_CONTAINER"
echo "Container ready: $main_image"
