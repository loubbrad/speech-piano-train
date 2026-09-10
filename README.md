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

## Build the Singularity or Apptainer image

The dependency image is separate because compiling `causal-conv1d` and FLA is
expensive. Adjust the generic build resources for the cluster, then run:

```bash
mkdir -p slurm
sbatch scripts/build_apptainer.sbatch --deps
```

Later source or config changes only require rebuilding the application image:

```bash
sbatch scripts/build_apptainer.sbatch
```

Image paths come from `config/local.yaml`.

## Prepare the two streams

Run preparation once per condition. The source dataset is mounted read-only and
the prepared-data directory is writable. Apptainer keeps its normal host-home
binding, so Hugging Face works without another configured path.

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
`execution.container_runtime` to the executable installed on the cluster.

## Submit training

```bash
speech-piano-submit qwen35-9b-interleaved \
    --config-file config/interleaved.yaml

speech-piano-submit qwen35-9b-separated \
    --config-file config/separated.yaml
```

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
