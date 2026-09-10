# Speech/Piano continued pretraining

This repository compares one pass of continued pretraining on the upstream
Speech/Piano interleaved documents with one pass on its separate speech and
MIDI documents. Both runs start from `Qwen/Qwen3.5-9B-Base`.

It contains no corpus construction, MIDI processing, custom model code,
Trainer integration, or Slurm watcher.

## Experiment

Only items marked `train` in the upstream document manifest are used:

- `interleaved` uses one `*.interleaved.doc.txt` document per video.
- `separated` treats `*.deinterleaved-speech.doc.txt` and nonempty
  `*.deinterleaved-midi.doc.txt` files as independent documents.

Preparation shuffles each condition once with a fixed seed, appends Qwen's EOS
token after every document, and writes one `uint32` token stream. A separated
ordering is reshuffled if speech and MIDI from the same YouTube ID are adjacent.
Training traverses the saved stream exactly once with `shuffle=False`.

Each prepared condition contains:

```text
<prepared-dir>/<variant>/
├── metadata.json
├── order.jsonl
└── tokens.bin
```

`order.jsonl` records the exact document order and token spans. Training drops
only the tail needed to form complete 4096-token sequences and optimizer
updates; those counts are recorded in `training.json`.

## Data

Authenticate and download the current upstream archive outside this repository:

```bash
hf auth login
hf download gclef-cmu/speech-piano \
    speech_piano_0.5.tar.gz \
    --type dataset \
    --local-dir /path/to/download
tar -xzf /path/to/download/speech_piano_0.5.tar.gz -C /path/to/data
```

`data.dataset_dir` should point to the extracted `speech_piano_0.5` directory.
Nothing else on the machine is inspected for Speech/Piano data.

## Configuration

Configuration is merged in this order:

1. `config/base.yaml`
2. `config/local.yaml`
3. The file passed through `--config-file`

Copy `config/local.example.yaml` to `config/local.yaml` and set the HPC paths
and Slurm directives. The local file is ignored by Git. Each submitted run
stores the merged configuration in its run directory.

The defaults target Qwen3.5 9B on four H100 80 GB GPUs with 4096-token
sequences, microbatch size 1, and 1,048,576 tokens per optimizer update. This
gives 64 gradient accumulation steps. Training uses a peak learning rate of
3e-5 with 5% warmup and cosine decay, and clips the gradient norm to 1.0 only
at optimizer-update boundaries.

## Dependencies

The tested PyTorch, Transformers, Accelerate, causal-conv1d, and FLA versions
are pinned, and `uv.lock` fixes the complete environment. Transformers supplies
Qwen, Accelerate supplies FSDP and checkpointing, and the CUDA extra supplies
Qwen3.5's fast DeltaNet kernels.

Local checks do not download the dataset or model:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

## Build the container

The Docker image is the single container build artifact. Its dependency layer
caches the expensive `causal-conv1d` and FLA compilation when only source or
configuration files change:

```bash
docker build -f containers/Dockerfile -t speech-piano:dev .
```

Test the image directly where Docker and the NVIDIA Container Toolkit are
available:

```bash
docker run --rm --gpus all speech-piano:dev \
    python -c 'import causal_conv1d, speech_piano_train, torch; print(torch.cuda.device_count())'
```

Convert that exact local image for an Apptainer or Singularity cluster:

```bash
apptainer build --force /path/to/speech-piano.sif \
    docker-daemon:speech-piano:dev
```

Alternatively, push an immutable tag to a registry for a Pyxis/Enroot cluster:

```bash
docker tag speech-piano:dev registry.example/speech-piano:GIT_SHA
docker push registry.example/speech-piano:GIT_SHA
```

Set `execution.container_runtime` to `apptainer`, `singularity`, or `pyxis` in
`config/local.yaml`. Apptainer and Singularity use a local SIF path. Pyxis uses
an Enroot registry reference such as
`registry.example#speech-piano:GIT_SHA`; it can also use an absolute path to a
pre-imported `.sqsh` image.

## Prepare the two streams

Run preparation once per condition. The source dataset is mounted read-only and
the prepared-data directory is writable. For Apptainer or Singularity:

```bash
for condition in interleaved separated; do
    singularity exec \
        --bind "$PWD/config/local.yaml:/workspace/speech-piano-train/config/local.yaml:ro" \
        --bind /path/to/data:/path/to/data:ro \
        --bind /path/to/speech-piano-prepared:/path/to/speech-piano-prepared \
        /path/to/speech-piano.sif \
        speech-piano-prepare \
        --config-file "/workspace/speech-piano-train/config/$condition.yaml"
done
```

Preparation fails if the condition directory already exists. Set
`execution.container_runtime` to the backend installed on the cluster.

With Pyxis/Enroot, the equivalent preparation command is:

```bash
for condition in interleaved separated; do
    srun \
        --container-image='registry.example#speech-piano:GIT_SHA' \
        --container-mounts="$PWD/config/local.yaml:/workspace/speech-piano-train/config/local.yaml:ro,/path/to/data:/path/to/data:ro,/path/to/speech-piano-prepared:/path/to/speech-piano-prepared" \
        --container-workdir=/workspace/speech-piano-train \
        --container-mount-home \
        speech-piano-prepare --config-file "config/$condition.yaml"
done
```

## Submit training

```bash
speech-piano-submit qwen35-9b-interleaved \
    --config-file config/interleaved.yaml

speech-piano-submit qwen35-9b-separated \
    --config-file config/separated.yaml
```

For Pyxis, the environment file configured by `execution.environment_file`
must use shell-compatible `KEY=value` lines. The generated batch script exports
those values before `srun`; Pyxis propagates them into the container.

The command creates a run directory, snapshots the configuration, writes
`job.sh`, submits it, and exits. Use `--dry-run` to generate the job without
calling `sbatch`.

Training uses BF16 FSDP full sharding, gradient checkpointing, AdamW, and a
token-based effective batch. Only the newest sharded checkpoint is retained.
At completion it is merged into a normal Hugging Face directory under `final/`.

Before full runs, use an interactive four-GPU allocation to confirm that the
container imports `causal_conv1d`, FLA, and Qwen3.5; all four H100s are visible;
one optimizer update fits; and a sharded checkpoint resumes and merges. Keep
model, tokenizer, optimizer, batch, and scheduler settings identical between
the two conditions.
