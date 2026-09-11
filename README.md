# Speech/Piano continued pretraining

This repository compares one pass of continued pretraining on the upstream
Speech/Piano interleaved documents with one pass on its separate speech and
MIDI documents. Both runs start from `Qwen/Qwen3.5-9B-Base`.

## Irmak HPC quick start

The committed experiment configurations target Irmak's eight-H100 Slurm
cluster. After cloning the repository to
`/home/ibukey/speech-piano-train`, create an untracked `.env`:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

Protect it and run the three setup/submission commands from the repository
root:

```bash
chmod 600 .env
./scripts/irmak/refresh-container.sh
./scripts/irmak/prepare-data.sh
./scripts/irmak/train.sh
```

`refresh-container.sh` imports the GHCR image tagged with the checked-out Git
commit, saves it under `/project/flame/ibukey/containers`, and validates it on
eight GPUs. Consequently, pull the repository before refreshing the image.
The public GHCR package must contain a successful image for that exact commit.

`prepare-data.sh` uses the container's `hf` command to download both the corpus
and Qwen checkpoint. The downloads, extracted data, model, Hugging Face cache,
and prepared token streams are explicit writable host mounts under
`/project/flame/ibukey`; they do not live in the container filesystem. The
script is safe to rerun after successful stages and reports partial output for
manual inspection.

`train.sh` creates a small Conda submission environment on first use and
submits only the interleaved condition. The separated configuration remains
available for a later explicit submission:

```bash
./scripts/irmak/train.sh interleaved
./scripts/irmak/train.sh separated
./scripts/irmak/train.sh --dry-run
```

Container images are published by `.github/workflows/container.yaml` on each
push to `main`. After its first run, ensure the GHCR package itself is public;
repository visibility and package visibility can be configured separately.
Non-secret paths and Slurm settings shared by the scripts are in
`scripts/irmak/common.sh`.

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

## Configuration

Configuration is merged in this order:

1. `config/base.yaml`
2. `config/local.yaml`, if present
3. The file passed through `--config-file`

The interleaved and separated files directly contain Irmak's paths, eight-GPU
allocation, and Slurm directives. Each submitted run stores the fully merged
configuration in its run directory.

Training uses 4096-token sequences, microbatch size 1, and 1,048,576 tokens per
optimizer update. On eight GPUs this gives 32 gradient accumulation steps. It
uses a peak learning rate of 3e-5 with 5% warmup and cosine decay, and clips the
gradient norm to 1.0 only at optimizer-update boundaries.

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

## Publish the container

The GitHub workflow publishes `linux/amd64` images to GHCR with both the full
Git commit and `main` tags. The dependency layer caches the expensive
`causal-conv1d` and FLA compilation. If the hosted builder is unavailable, the
same image can be published manually from an x86-64 Docker machine:

```bash
revision=$(git rev-parse HEAD)
image=ghcr.io/loubbrad/speech-piano-train:$revision
docker build --platform linux/amd64 \
    --build-arg TORCH_CUDA_ARCH_LIST=9.0 \
    --build-arg VCS_REF="$revision" \
    -f containers/Dockerfile -t "$image" .
docker push "$image"
```

Training uses BF16 FSDP full sharding, gradient checkpointing, AdamW, and a
token-based effective batch. Only the newest sharded checkpoint is retained.
At completion it is merged into a normal Hugging Face directory under `final/`.

The refresh script checks imports, all eight H100s, and the NVLink topology.
Before committing to both full runs, it is still prudent to confirm that one
optimizer update fits and that a sharded checkpoint resumes and merges. Keep
model, tokenizer, optimizer, batch, and scheduler settings identical between
the two conditions.
