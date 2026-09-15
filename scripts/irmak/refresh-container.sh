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

logs_dir="$repo_dir/logs"
mkdir -p "$logs_dir"
export IRMAK_IMAGE_URI="$image_uri"
export IRMAK_PARTIAL_IMAGE="$partial"
export IRMAK_MAIN_IMAGE="$main_image"

irmak_sbatch_args
echo "Importing $image_uri"
echo "Slurm output: $logs_dir/refresh-container-<job-id>.out"
sbatch --wait \
    "${IRMAK_SBATCH_ARGS[@]}" \
    --job-name=refresh-container \
    --chdir="$repo_dir" \
    --output="$logs_dir/refresh-container-%j.out" \
    --export=ALL \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --time=01:00:00 \
    <<'BATCH'
#!/usr/bin/env bash
set -euo pipefail

enroot import --output "$IRMAK_PARTIAL_IMAGE" "$IRMAK_IMAGE_URI"
mv -- "$IRMAK_PARTIAL_IMAGE" "$IRMAK_MAIN_IMAGE"
BATCH

ln -sfn -- "$(basename -- "$main_image")" "$IRMAK_CONTAINER"
echo "Container ready: $main_image"
