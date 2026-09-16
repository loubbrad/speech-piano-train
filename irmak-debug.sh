#!/usr/bin/env bash

# Collect the cluster details needed to diagnose the Irmak launch scripts.
#
# This script is intentionally best-effort: missing programs, restricted Slurm
# queries, and unavailable paths are reported but never make the script fail.
# It does not submit a job and never prints values from .env.

section() {
    printf '\n===== %s =====\n' "$1"
}

available() {
    command -v "$1" >/dev/null 2>&1
}

run() {
    local status

    printf '$'
    printf ' %q' "$@"
    printf '\n'

    if available timeout; then
        timeout 15 "$@" 2>&1
    else
        "$@" 2>&1
    fi
    status=$?

    if (( status != 0 )); then
        printf '[check unavailable or returned status %d; continuing]\n' "$status"
    fi
    return 0
}

command_location() {
    local name=$1
    local location

    if location=$(command -v "$name" 2>/dev/null); then
        printf '%-12s %s\n' "$name" "$location"
    else
        printf '%-12s MISSING\n' "$name"
    fi
}

container_options() {
    local command_name=$1
    local help_text

    if ! available "$command_name"; then
        printf '%s is missing; cannot inspect its Pyxis options\n' "$command_name"
        return 0
    fi

    if available timeout; then
        help_text=$(timeout 15 "$command_name" --help 2>&1) || true
    else
        help_text=$("$command_name" --help 2>&1) || true
    fi

    if ! grep -E -- '--container-(image|mounts|workdir|env)' <<<"$help_text"; then
        printf 'No Pyxis container options were shown by %s --help\n' "$command_name"
    fi
    return 0
}

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)
if [[ -z ${repo_dir:-} || ! -d $repo_dir ]]; then
    repo_dir=$PWD
fi

section "Report metadata"
run date -u
printf 'script=%s\n' "${BASH_SOURCE[0]}"
printf 'repository=%s\n' "$repo_dir"

section "Identity and host"
run hostname -f
run whoami
run id
run uname -a
printf 'shell=%s\n' "${SHELL:-unknown}"

section "Repository"
if available git && git -C "$repo_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    run git -C "$repo_dir" status --short
    run git -C "$repo_dir" log -1 --oneline
else
    printf '%s is not a Git working tree, or git is unavailable\n' "$repo_dir"
fi

section "Environment file"
environment_file=$repo_dir/.env
if [[ -f $environment_file ]]; then
    if available stat; then
        run stat -c 'permissions=%a owner=%U:%G file=%n' "$environment_file"
    else
        run ls -l "$environment_file"
    fi

    for variable in HF_TOKEN WANDB_API_KEY GHCR_TOKEN; do
        if LC_ALL=C grep -q "^${variable}=." "$environment_file" 2>/dev/null; then
            printf '%s is present (value hidden)\n' "$variable"
        else
            printf '%s is MISSING\n' "$variable"
        fi
    done
else
    printf '%s is MISSING\n' "$environment_file"
fi

section "Scheduler commands"
for command_name in sbatch srun salloc squeue sinfo scontrol sacct sacctmgr; do
    command_location "$command_name"
done

section "Slurm client and controller"
if available sbatch; then
    run sbatch --version
fi
if available srun; then
    run srun --version
fi
if available scontrol; then
    run scontrol ping
fi
printf 'SLURM_CONF=%s\n' "${SLURM_CONF:-not set}"
printf 'SLURM_CLUSTER_NAME=%s\n' "${SLURM_CLUSTER_NAME:-not set}"

section "Partitions and GPUs"
if available sinfo; then
    run sinfo -o '%P %a %l %D %G'
else
    printf 'sinfo is missing\n'
fi

section "User Slurm associations"
if available sacctmgr; then
    run sacctmgr -nP show assoc where "user=${USER:-}" \
        format=Cluster,Account,User,Partition,QOS
else
    printf 'sacctmgr is missing; this query is often restricted on clusters\n'
fi

section "Pyxis support"
container_options srun
container_options sbatch

section "Container runtimes"
for command_name in enroot apptainer singularity docker podman; do
    command_location "$command_name"
done
if available enroot; then
    run enroot version
fi
if available apptainer; then
    run apptainer --version
fi
if available singularity; then
    run singularity --version
fi

section "Project storage"
project_root=/project/flame/ibukey
if [[ -e $project_root ]]; then
    run ls -ld "$project_root"
    run df -h "$project_root"
    [[ -r $project_root ]] && readable=yes || readable=no
    [[ -w $project_root ]] && writable=yes || writable=no
    printf 'readable=%s writable=%s path=%s\n' "$readable" "$writable" "$project_root"
else
    printf '%s does not exist on this host\n' "$project_root"
fi
if available quota; then
    run quota -s
fi

section "Expected container"
container=/project/flame/ibukey/containers/speech-piano-current.sqsh
if [[ -f $container ]]; then
    run ls -lh "$container"
else
    printf '%s does not exist yet\n' "$container"
fi

section "Configured Irmak values"
common_file=$repo_dir/scripts/irmak/common.sh
if [[ -f $common_file ]]; then
    grep -E \
        '^(IRMAK_ACCOUNT|IRMAK_PARTITION|IRMAK_QOS|IRMAK_PROJECT_ROOT)=' \
        "$common_file" 2>/dev/null || true
else
    printf '%s is missing\n' "$common_file"
fi

section "Finished"
printf 'Diagnostics completed. No jobs were submitted.\n'
printf 'Missing commands and unsuccessful checks above are diagnostic results, not script failures.\n'
printf 'Send this report back, but do not send the contents of .env.\n'

exit 0
