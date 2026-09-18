# Speech/Piano continued pretraining

This repository runs three one-pass continued-pretraining experiments from
`Qwen/Qwen3.5-9B-Base`: interleaved MIDI text, separated speech and MIDI text,
and interleaved speech with MIDI rendered online as piano audio.

## Irmak HPC quick start

The committed experiment configurations target Irmak's eight-H100 Flame
nodes. The checkout is `/home/ibukey/louis-speech-piano-train`. Create an
untracked `.env` there:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=...
PIANOTEQ_KEY=...
GHCR_TOKEN=ghp_...
# GHCR_USERNAME=your-github-username
```

Protect it and check the cluster environment from the repository root:

```bash
chmod 600 .env
./scripts/irmak/check-environment.sh
```

After the GitHub container workflow for the current `main` commit has
finished successfully, run:

```bash
./scripts/irmak/refresh-container.sh
./scripts/irmak/prepare-data.sh
./scripts/irmak/train.sh --dry-run > /tmp/speech-piano-jobs.txt
./scripts/irmak/train.sh all
```

`refresh-container.sh` runs on the login node, imports the latest `main` image
from GHCR, and saves it under `/project/flame/ibukey/containers`. `GHCR_TOKEN`
needs `read:packages` access to the package. Enroot's layer cache stays on the
project filesystem. Layer extraction uses a unique directory under
`/dev/shm/$USER/speech-piano-enroot` because the project filesystem is NFS and
does not support Enroot's overlay xattrs. Successful imports remove that
temporary directory.

`prepare-data.sh` also runs on the login node and starts the imported image with
Enroot. It uses the container's `hf` command to download both the corpus and
Qwen checkpoint. It uses the existing 0.4 corpus at
`/project/flame/ibukey/speech_piano_data/speech_piano_0.4` and prepares these
three streams:

```text
/project/flame/ibukey/speech-piano-prepared/interleaved/midi_text
/project/flame/ibukey/speech-piano-prepared/separated/midi_text
/project/flame/ibukey/speech-piano-prepared/interleaved/mel
```

Downloads, the model, Hugging Face cache, and prepared streams are persistent
host paths under `/project/flame/ibukey`; they do not live in the container.
The script is safe to rerun after completed stages and reports incomplete
output for manual inspection.

`train.sh` creates a small Conda submission environment on first use. The
commands for individual eight-GPU runs are:

```bash
./scripts/irmak/train.sh interleaved
./scripts/irmak/train.sh separated
./scripts/irmak/train.sh interleaved-mel
```

`all` submits all three. `--dry-run` prints all three batch scripts without
creating experiment directories or submitting jobs. The run directories and
W&B names are `qwen35-9b-interleaved-midi`, `qwen35-9b-separated-midi`, and
`qwen35-9b-interleaved-mel`. Only the mel job activates Pianoteq and requires
`PIANOTEQ_KEY` at runtime.

Container images are published by `.github/workflows/container.yaml` on each
push to `main`.
Non-secret paths and Slurm settings shared by the scripts are in
`scripts/irmak/common.sh`.

## Experiment

Only items marked `train` in the upstream document manifest are used:

- `interleaved` uses one `*.interleaved.doc.txt` document per video.
- `separated` treats `*.deinterleaved-speech.doc.txt` and nonempty
  `*.deinterleaved-midi.doc.txt` files as independent documents.

Preparation shuffles each condition once with a fixed seed and appends Qwen's
EOS token after every document. A separated ordering is reshuffled if speech
and MIDI from the same YouTube ID are adjacent. The `midi_text` representation
stores ordinary text tokens; the interleaved-only `mel` representation stores
text tokens plus serialized piano blocks for online Pianoteq rendering.

Each prepared condition contains:

```text
<prepared-dir>/<variant>/<representation>/
├── metadata.json
├── order.jsonl
├── tokens.bin
├── piano_blocks.bin
├── segments.bin
└── index.bin
```

`order.jsonl` records the document order and logical spans. Both representations
use the same indexed logical-stream reader. Audio DataLoader workers render MIDI
with Pianoteq, convert it to 100 Hz log-mel features, and may return examples out of
order. Training drops only the tails needed to form complete 4096-position
sequences and optimizer updates; those counts are recorded in `training.json`.

## Configuration

Configuration is merged in this order:

1. `config/base.yaml`
2. `config/local.yaml`, if present
3. The file passed through `--config-file`

The two text files and the interleaved-mel file directly contain Irmak's
paths, eight-GPU allocation, and Slurm directives. Each submitted run stores
the fully merged configuration in its run directory. The interleaved-mel run
uses 16 DataLoader workers per GPU process; the text runs use two.

Training uses 4096-token sequences, microbatch size 1, and 2,097,152 tokens per
optimizer update. On eight GPUs this gives 64 gradient accumulation steps. It
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
same image can be published manually from an x86-64 Docker machine. Put a
classic GitHub token with `write:packages` access in the ignored `.env` file:

```dotenv
GHCR_TOKEN=ghp_your_classic_token_here
```

Then run `./scripts/publish-container.sh`. `GHCR_USERNAME` and
`GHCR_IMAGE_REPOSITORY` can also be set in `.env` to override their defaults.
The script builds and pushes both tags directly with Buildx.

Training uses BF16 FSDP full sharding, gradient checkpointing, AdamW, and a
token-based effective batch. Only the newest sharded checkpoint is retained.
At completion it is merged into a normal Hugging Face directory under `final/`.

Each training job checks that all eight requested GPUs are visible. Before
committing to the full audio run, confirm that Pianoteq activates on a compute
node and that one optimizer update fits. Also confirm that a sharded checkpoint
resumes and merges before relying on the long runs.
