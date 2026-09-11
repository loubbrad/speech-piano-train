#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
env_file="$repo_dir/.env"

if [[ ! -f $env_file ]]; then
    echo "Missing $env_file; add GHCR_TOKEN with write:packages access" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$env_file"

registry_username=${GHCR_USERNAME:-loubbrad}
registry_token=${GHCR_TOKEN:-${CR_PAT:-${GH_TOKEN:-${GITHUB_TOKEN:-}}}}
image_repository=${GHCR_IMAGE_REPOSITORY:-ghcr.io/loubbrad/speech-piano-train}

if [[ -z $registry_token ]]; then
    echo "Set GHCR_TOKEN (or CR_PAT/GH_TOKEN/GITHUB_TOKEN) in $env_file" >&2
    exit 1
fi

for command in docker git; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        exit 1
    fi
done

if ! docker buildx version >/dev/null 2>&1; then
    echo "Docker Buildx is required but unavailable" >&2
    exit 1
fi

cd "$repo_dir"
revision=$(git rev-parse HEAD)

printf '%s' "$registry_token" | \
    docker login ghcr.io --username "$registry_username" --password-stdin
unset registry_token GHCR_TOKEN CR_PAT GH_TOKEN GITHUB_TOKEN

docker buildx build \
    --platform linux/amd64 \
    --file containers/Dockerfile \
    --build-arg TORCH_CUDA_ARCH_LIST=9.0 \
    --build-arg CUDA_BUILD_JOBS=2 \
    --build-arg VCS_REF="$revision" \
    --tag "$image_repository:$revision" \
    --tag "$image_repository:main" \
    --push \
    .
