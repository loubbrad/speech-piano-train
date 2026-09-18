# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS dependencies

ARG TORCH_CUDA_ARCH_LIST=9.0
ARG CUDA_BUILD_JOBS=8
ARG PIANOTEQ_URL=https://drive.google.com/uc?export=download&id=17T4gqbwcrl7oHGsaIjKLF3vw2S3UzhCO
ARG PIANOTEQ_SHA256=7cd54766c9482cb543bb9cf7f4c9bb5b11e32fc038bf441adccc9eebb171dcc8

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/speech-piano/.venv/bin:/usr/local/bin:$PATH \
    UV_PROJECT_ENVIRONMENT=/opt/speech-piano/.venv \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update \
    && apt-get install -y -o Dpkg::Use-Pty=0 --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libasound2t64 \
        libfreetype6 \
        libgl1 \
        p7zip-full \
        python3 \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin sh

RUN curl -fL --retry 3 -o /tmp/pianoteq.7z "$PIANOTEQ_URL" \
    && echo "$PIANOTEQ_SHA256  /tmp/pianoteq.7z" | sha256sum -c - \
    && mkdir -p /opt/pianoteq \
    && 7z e -y -o/opt/pianoteq /tmp/pianoteq.7z \
        'Pianoteq 8 STAGE/x86-64bit/Pianoteq 8 STAGE' \
    && chmod 0755 '/opt/pianoteq/Pianoteq 8 STAGE' \
    && ln -s '/opt/pianoteq/Pianoteq 8 STAGE' '/usr/local/bin/Pianoteq 8 STAGE' \
    && 'Pianoteq 8 STAGE' --version \
    && rm /tmp/pianoteq.7z

WORKDIR /workspace/speech-piano-train

COPY pyproject.toml uv.lock ./
RUN MAX_JOBS=${CUDA_BUILD_JOBS} CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    uv sync --frozen --no-dev --extra cuda-ext --no-install-project \
        --no-binary-package causal-conv1d \
    && uv cache clean

FROM dependencies AS application

ARG VCS_REF=unknown

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y -o Dpkg::Use-Pty=0 --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY speech_piano_train ./speech_piano_train
COPY config/accelerate-fsdp.yaml config/base.yaml \
    config/interleaved.yaml config/interleaved-mel.yaml \
    config/separated.yaml ./config/

RUN uv sync --frozen --no-dev --extra cuda-ext \
    && uv cache clean \
    && chmod -R a+rX /workspace/speech-piano-train \
    && printf '%s\n' "$VCS_REF" > /opt/speech-piano/image-revision \
    && python -c \
        "import causal_conv1d, speech_piano_train, torch, torchcodec; from speech_piano_train.data import TokenDataset; from transformers import Qwen3_5ForCausalLM" \
    && hf version
