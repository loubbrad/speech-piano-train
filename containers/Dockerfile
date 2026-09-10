# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS dependencies

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/speech-piano/.venv/bin:/usr/local/bin:$PATH \
    UV_PROJECT_ENVIRONMENT=/opt/speech-piano/.venv

RUN apt-get update \
    && apt-get install -y -o Dpkg::Use-Pty=0 --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /workspace/speech-piano-train

COPY pyproject.toml uv.lock ./
RUN MAX_JOBS=8 CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    uv sync --frozen --no-dev --extra cuda-ext --no-install-project \
        --no-binary-package causal-conv1d \
    && uv cache clean

FROM dependencies AS application

ENV PYTHONUNBUFFERED=1

COPY speech_piano_train ./speech_piano_train
COPY config/accelerate-fsdp.yaml config/base.yaml \
    config/interleaved.yaml config/separated.yaml ./config/

RUN uv sync --frozen --no-dev --extra cuda-ext \
    && uv cache clean \
    && chmod -R a+rX /workspace/speech-piano-train \
    && python -c \
        "import causal_conv1d, speech_piano_train, torch; from transformers import Qwen3_5ForCausalLM"
