#!/usr/bin/env bash

# Shared, non-secret settings for Irmak's cluster.
IRMAK_ACCOUNT=chrisdon
IRMAK_PARTITION=flame
IRMAK_QOS=flame-16gpu_qos

IRMAK_PROJECT_ROOT=/project/flame/ibukey
IRMAK_CONTAINER_DIR=$IRMAK_PROJECT_ROOT/containers
IRMAK_CONTAINER=$IRMAK_CONTAINER_DIR/speech-piano-current.sqsh
IRMAK_DOWNLOAD_DIR=$IRMAK_PROJECT_ROOT/downloads
IRMAK_DATA_ROOT=$IRMAK_PROJECT_ROOT/data
IRMAK_DATASET_DIR=$IRMAK_DATA_ROOT/speech_piano_0.5
IRMAK_MODEL_ROOT=$IRMAK_PROJECT_ROOT/models
IRMAK_MODEL_DIR=$IRMAK_MODEL_ROOT/Qwen3.5-9B-Base
IRMAK_PREPARED_ROOT=$IRMAK_PROJECT_ROOT/speech-piano-prepared
IRMAK_HF_HOME=$IRMAK_PROJECT_ROOT/huggingface

IRMAK_IMAGE_REPOSITORY=ghcr.io/loubbrad/speech-piano-train
IRMAK_DATA_REPOSITORY=gclef-cmu/speech-piano
IRMAK_DATA_ARCHIVE=speech_piano_0.5.tar.gz
IRMAK_MODEL_REPOSITORY=Qwen/Qwen3.5-9B-Base

irmak_srun_args() {
    IRMAK_SRUN_ARGS=(
        --account="$IRMAK_ACCOUNT"
        --partition="$IRMAK_PARTITION"
        --nodes=1
        --ntasks=1
    )
    if [[ -n $IRMAK_QOS ]]; then
        IRMAK_SRUN_ARGS+=(--qos="$IRMAK_QOS")
    fi
}
