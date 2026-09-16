#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=common.sh
source "$repo_dir/scripts/irmak/common.sh"
cd "$repo_dir"

# Accept --local for backward compatibility. This script now always imports on
# the current node (upstream removed the Slurm path), so the flag is a no-op.
for arg in "$@"; do
    case "$arg" in
        --local) ;;
        *) echo "Unknown argument: $arg (only --local is supported)" >&2; exit 1 ;;
    esac
done

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

# enroot's layer extraction + whiteout conversion must run on a LOCAL,
# xattr-capable filesystem: it sets overlayfs "opaque" xattrs that NFS
# (/project/flame here) rejects with "failed to create opaque ovlfs whiteout:
# ... Not supported". The default /tmp is a tmpfs (xattrs work) but small and
# full, and /mnt/tmp is not user-writable, so use /dev/shm (also tmpfs, so
# xattrs work, with more room). Override with ENROOT_SCRATCH_DIR. The layer
# cache is plain blobs (no xattrs), so it can stay on the project filesystem.
enroot_scratch="${ENROOT_SCRATCH_DIR:-/dev/shm/$USER/enroot}"
export ENROOT_CACHE_PATH="$IRMAK_CONTAINER_DIR/enroot-cache"
export ENROOT_DATA_PATH="$enroot_scratch/data"
export ENROOT_TEMP_PATH="$enroot_scratch/tmp"
export TMPDIR="$ENROOT_TEMP_PATH"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH"

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
