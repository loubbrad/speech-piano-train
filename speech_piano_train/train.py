from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import merge_fsdp_weights, set_seed
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import Qwen3_5ForCausalLM, get_cosine_schedule_with_warmup

from speech_piano_train.config import AppConfig
from speech_piano_train.data import TokenDataset


@dataclass
class TrainProgress:
    batch_cursor: int = 0
    global_update: int = 0

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.batch_cursor = state["batch_cursor"]
        self.global_update = state["global_update"]


def train(config: AppConfig, run_dir: str | Path) -> None:
    run_dir = Path(run_dir).resolve()
    accelerator = Accelerator(
        project_dir=run_dir,
        log_with="wandb",
    )
    if accelerator.mixed_precision != "bf16":
        raise ValueError("Training requires Accelerate BF16 mixed precision")
    if accelerator.distributed_type != DistributedType.FSDP:
        raise ValueError("Training requires Accelerate FSDP")

    dataset = TokenDataset(config.data.prepared_path)
    divisor = accelerator.num_processes * config.train.micro_batch_size
    update_tokens = dataset.sequence_length * divisor
    if config.train.tokens_per_update % update_tokens:
        raise ValueError(
            "tokens_per_update must be divisible by "
            "sequence_length * world_size * micro_batch_size"
        )
    accumulation_steps = config.train.tokens_per_update // update_tokens
    sequences_per_update = config.train.tokens_per_update // dataset.sequence_length
    usable_sequences = len(dataset) // sequences_per_update * sequences_per_update
    if usable_sequences == 0:
        raise ValueError("Prepared data has no complete optimizer update")
    accelerator.gradient_accumulation_steps = accumulation_steps

    dataloader = DataLoader(
        Subset(dataset, range(usable_sequences)),
        batch_size=config.train.micro_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=config.train.num_workers > 0,
    )

    set_seed(config.train.seed)
    model = Qwen3_5ForCausalLM.from_pretrained(
        config.model.name,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
        weight_decay=config.train.weight_decay,
    )
    model, optimizer, dataloader = accelerator.prepare(
        model,
        optimizer,
        dataloader,
    )
    total_updates = len(dataloader) // accumulation_steps
    warmup_updates = round(total_updates * config.train.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
    )

    training_config = {
        "config": config.model_dump(mode="json"),
        "dataset": dataset.metadata,
        "world_size": accelerator.num_processes,
        "usable_sequences": usable_sequences,
        "unused_sequences": len(dataset) - usable_sequences,
        "accumulation_steps": accumulation_steps,
        "total_updates": total_updates,
    }
    setup_run(accelerator, run_dir, training_config)

    progress = TrainProgress()
    accelerator.register_for_checkpointing(progress)
    accelerator.register_for_checkpointing(scheduler)
    checkpoints = run_dir / "checkpoints"
    saved = sorted(checkpoints.glob("checkpoint-*"))
    latest = saved[-1] if saved else None
    if latest is not None:
        accelerator.print(f"Resuming from {latest}")
        accelerator.load_state(str(latest))
    accelerator.init_trackers(
        config.wandb.project,
        config=training_config,
        init_kwargs={
            "wandb": {
                "name": config.wandb.name or run_dir.name,
                "mode": config.wandb.mode,
                "dir": str(run_dir),
            }
        },
    )
    accelerator.print(
        f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    accelerator.print(
        f"Sequences: {usable_sequences:,}; updates: {total_updates:,}; "
        f"accumulation: {accumulation_steps}"
    )

    remaining = accelerator.skip_first_batches(
        dataloader,
        progress.batch_cursor,
    )
    batches = tqdm(
        remaining,
        total=len(dataloader) - progress.batch_cursor,
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )
    update_started = time.perf_counter()
    accumulated_loss = torch.zeros((), device=accelerator.device)
    accumulated_batches = 0

    model.train()
    for batch_index, input_ids in enumerate(
        batches,
        start=progress.batch_cursor,
    ):
        with accelerator.accumulate(model):
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            loss = output.loss
            accumulated_loss += loss.detach()
            accumulated_batches += 1
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(),
                    config.train.max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad()

        if not accelerator.sync_gradients:
            continue
        progress.batch_cursor = batch_index + 1
        progress.global_update += 1
        scheduler.step()
        mean_loss = accelerator.reduce(accumulated_loss, reduction="mean")
        mean_loss /= accumulated_batches
        elapsed = time.perf_counter() - update_started
        update_token_count = accumulated_batches * update_tokens
        metrics = {
            "train/loss": mean_loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/tokens_per_second": update_token_count / elapsed,
        }
        accelerator.log(metrics, step=progress.global_update)
        accumulated_loss.zero_()
        accumulated_batches = 0
        update_started = time.perf_counter()

        if progress.global_update % config.train.checkpoint_interval == 0:
            save_checkpoint(accelerator, run_dir, progress)

    checkpoint = checkpoints / f"checkpoint-{progress.global_update:08d}"
    if not checkpoint.is_dir():
        checkpoint = save_checkpoint(accelerator, run_dir, progress)
    export_model(accelerator, model, config, run_dir, checkpoint)
    accelerator.end_training()


def setup_run(
    accelerator: Accelerator,
    run_dir: Path,
    training_config: dict[str, Any],
) -> None:
    path = run_dir / "training.json"
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        path.write_text(
            json.dumps(training_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()


def save_checkpoint(
    accelerator: Accelerator,
    run_dir: Path,
    progress: TrainProgress,
) -> Path:
    checkpoints = run_dir / "checkpoints"
    name = f"checkpoint-{progress.global_update:08d}"
    temporary = checkpoints / f".{name}.tmp"
    output = checkpoints / name
    if accelerator.is_main_process:
        shutil.rmtree(temporary, ignore_errors=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(temporary), safe_serialization=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        temporary.rename(output)
        completed = sorted(checkpoints.glob("checkpoint-*"))
        for old in completed[:-1]:
            shutil.rmtree(old)
    accelerator.wait_for_everyone()
    return output


def export_model(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: AppConfig,
    run_dir: Path,
    checkpoint: Path,
) -> None:
    output = run_dir / "final"
    if accelerator.is_main_process:
        output.mkdir(exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.config.save_pretrained(output)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name,
        )
        tokenizer.save_pretrained(output)
    accelerator.wait_for_everyone()

    merge_fsdp_weights(
        str(checkpoint / "pytorch_model_fsdp_0"),
        str(output),
        safe_serialization=True,
    )
