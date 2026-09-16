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

partial="$main_image.partial"
if [[ -e $partial ]]; then
    echo "Remove the incomplete image before retrying: $partial" >&2
    exit 1
fi

echo "Importing $image_uri"
enroot import --output "$partial" "$image_uri"
mv -- "$partial" "$main_image"

ln -sfn -- "$(basename -- "$main_image")" "$IRMAK_CONTAINER"
echo "Container ready: $main_image"
