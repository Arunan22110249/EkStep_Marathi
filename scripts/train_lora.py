from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model

from marathi_tts.config import TrainingConfig
from marathi_tts.training import create_training_dataset


MODEL_NAME = "bodhan-ai/indic-speak"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for Bodhan Marathi TTS"
    )

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)

    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )

    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoints",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="lora_run",
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device_and_dtype(
    requested_dtype: str,
) -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        device = torch.device("cuda")

        if requested_dtype == "float32":
            dtype = torch.float32

        elif requested_dtype == "float16":
            dtype = torch.float16

        elif requested_dtype == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "bfloat16 was requested, but this CUDA device "
                    "does not support bfloat16."
                )
            dtype = torch.bfloat16

        else:
            # Prefer BF16 when supported, otherwise FP16.
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

        return device, dtype

    # CPU training is only intended for smoke tests.
    return torch.device("cpu"), torch.float32


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig()

    if args.max_samples is not None:
        config.max_samples = args.max_samples

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.gradient_accumulation is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs

    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps

    if args.save_steps is not None:
        config.save_steps = args.save_steps

    if args.logging_steps is not None:
        config.logging_steps = args.logging_steps

    return config


def save_checkpoint(
    checkpoint_dir: Path,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    global_step: int,
    micro_step: int,
    config: TrainingConfig,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # PEFT saves only the LoRA adapter, not the 3B+ base model.
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "global_step": global_step,
            "micro_step": micro_step,
        },
        checkpoint_dir / "trainer_state.pt",
    )

    metadata = {
        "model": MODEL_NAME,
        "global_step": global_step,
        "micro_step": micro_step,
        "training_config": asdict(config),
        "arguments": vars(args),
    }

    with open(checkpoint_dir / "run_state.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    device, dtype = select_device_and_dtype(args.dtype)

    config = build_training_config(args)

    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Bodhan Marathi TTS LoRA Fine-Tuning")
    print("=" * 80)
    print(f"Model:              {MODEL_NAME}")
    print(f"Device:             {device}")
    print(f"Dtype:              {dtype}")
    print(f"CUDA available:     {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:                {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory:         "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(f"Dataset:            {config.dataset_name}")
    print(f"Max samples:        {config.max_samples}")
    print(f"Sequence length:    {config.max_seq_length}")
    print(f"Batch size:         {config.batch_size}")
    print(
        f"Grad accumulation:  "
        f"{config.gradient_accumulation_steps}"
    )
    print(f"Learning rate:      {config.learning_rate}")
    print(f"LoRA rank:          {config.lora_r}")
    print(f"LoRA alpha:         {config.lora_alpha}")
    print(f"LoRA targets:       {config.lora_target_modules}")
    print(f"Output:             {run_dir}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print(f"Tokenizer base vocab: {tokenizer.vocab_size}")
    print(f"Tokenizer total size: {len(tokenizer)}")
    print(f"Pad token ID:         {tokenizer.pad_token_id}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    print("\nPreparing dataset...")

    dataset, collator, _ = create_training_dataset(
        config,
        tokenizer,
    )

    print(f"Dataset samples: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError("Dataset contains zero training examples.")

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": True,
        "collate_fn": collator,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }

    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    dataloader = DataLoader(**loader_kwargs)

    print(f"DataLoader batches/epoch: {len(dataloader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print("\nLoading Bodhan model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model.to(device)

    # ------------------------------------------------------------------
    # Memory optimizations
    # ------------------------------------------------------------------

    if device.type == "cuda":
        model.config.use_cache = False

        try:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            print("Gradient checkpointing: enabled")
        except Exception as exc:
            print(
                "WARNING: Could not enable gradient checkpointing: "
                f"{exc}"
            )

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    print("\nAttaching LoRA adapter...")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError("No trainable LoRA parameters were found.")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    optimizer = AdamW(
        trainable_parameters,
        lr=config.learning_rate,
    )

    # Determine number of optimizer steps.
    steps_per_epoch = math.ceil(
        len(dataloader) / config.gradient_accumulation_steps
    )

    if args.max_steps is not None:
        total_optimizer_steps = args.max_steps
        training_epochs = math.ceil(total_optimizer_steps / steps_per_epoch)
    else:
        total_optimizer_steps = steps_per_epoch * config.num_epochs
        training_epochs = config.num_epochs

    if total_optimizer_steps <= 0:
        raise RuntimeError("Calculated zero optimizer steps.")

    warmup_steps = min(
        config.warmup_steps,
        total_optimizer_steps,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    # ------------------------------------------------------------------
    # Mixed precision
    # ------------------------------------------------------------------

    use_cuda_autocast = device.type == "cuda"

    use_grad_scaler = (
        device.type == "cuda"
        and dtype == torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_grad_scaler,
    )

    print(f"Mixed precision autocast: {use_cuda_autocast}")
    print(f"Gradient scaler:          {use_grad_scaler}")
    print(f"Total optimizer steps:    {total_optimizer_steps}")
    print(f"Warmup steps:             {warmup_steps}")

    # ------------------------------------------------------------------
    # Save run configuration
    # ------------------------------------------------------------------

    run_config = {
        "model": MODEL_NAME,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
        "training_config": asdict(config),
        "arguments": vars(args),
        "dataset_size": len(dataset),
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "warmup_steps": warmup_steps,
        "gradient_checkpointing": (
            device.type == "cuda"
        ),
    }

    with open(
        run_dir / "run_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_config,
            f,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    print("\nStarting training...")
    print("=" * 80)

    model.train()

    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    micro_step = 0
    accumulated_loss = 0.0
    loss_history: list[dict[str, float]] = []

    start_time = time.time()

    stop_training = False

    for epoch in range(training_epochs):
        if stop_training:
            break

        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        for batch_index, batch in enumerate(dataloader):
            micro_step += 1

            input_ids = batch["input_ids"].to(
                device=device,
                dtype=torch.long,
                non_blocking=device.type == "cuda",
            )

            attention_mask = batch["attention_mask"].to(
                device=device,
                dtype=torch.long,
                non_blocking=device.type == "cuda",
            )

            labels = batch["labels"].to(
                device=device,
                dtype=torch.long,
                non_blocking=device.type == "cuda",
            )

            # ----------------------------------------------------------
            # Forward pass
            # ----------------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=use_cuda_autocast,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                loss_for_backward = (
                    loss / config.gradient_accumulation_steps
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss encountered at "
                    f"micro-step {micro_step}: {loss.item()}"
                )

            accumulated_loss += loss.item()

            # ----------------------------------------------------------
            # Backward
            # ----------------------------------------------------------

            if use_grad_scaler:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            should_step = (
                micro_step % config.gradient_accumulation_steps == 0
            )

            # Also step on the final incomplete accumulation group.
            is_last_batch = (
                batch_index == len(dataloader) - 1
            )

            if is_last_batch:
                should_step = True

            if not should_step:
                continue

            # ----------------------------------------------------------
            # Optimizer step
            # ----------------------------------------------------------

            if use_grad_scaler:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            if use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            average_loss = (
                accumulated_loss
                / config.gradient_accumulation_steps
            )

            elapsed = time.time() - start_time

            loss_history.append(
                {
                    "step": float(global_step),
                    "loss": float(average_loss),
                    "learning_rate": float(
                        scheduler.get_last_lr()[0]
                    ),
                    "elapsed_seconds": float(elapsed),
                }
            )

            if (
                global_step == 1
                or global_step % config.logging_steps == 0
            ):
                print(
                    f"step={global_step} "
                    f"micro_step={micro_step} "
                    f"loss={average_loss:.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"elapsed={elapsed:.1f}s"
                )

            accumulated_loss = 0.0

            # ----------------------------------------------------------
            # Checkpoint
            # ----------------------------------------------------------

            if (
                config.save_steps > 0
                and global_step % config.save_steps == 0
            ):
                checkpoint_dir = (
                    run_dir / f"checkpoint-{global_step}"
                )

                print(
                    f"Saving checkpoint: {checkpoint_dir}"
                )

                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=global_step,
                    micro_step=micro_step,
                    config=config,
                    args=args,
                )

            if (
                args.max_steps is not None
                and global_step >= args.max_steps
            ):
                stop_training = True
                break

    # ------------------------------------------------------------------
    # Final adapter
    # ------------------------------------------------------------------

    final_adapter_dir = run_dir / "final_adapter"

    print("\nSaving final LoRA adapter...")

    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # Save loss history.
    with open(
        run_dir / "loss_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            loss_history,
            f,
            indent=2,
        )

    elapsed_seconds = time.time() - start_time

    summary = {
        "model": MODEL_NAME,
        "device": str(device),
        "dtype": str(dtype),
        "dataset_samples": len(dataset),
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": (
            config.gradient_accumulation_steps
        ),
        "micro_steps": micro_step,
        "optimizer_steps": global_step,
        "elapsed_seconds": elapsed_seconds,
        "final_loss": (
            loss_history[-1]["loss"]
            if loss_history
            else None
        ),
        "final_learning_rate": (
            loss_history[-1]["learning_rate"]
            if loss_history
            else None
        ),
        "final_adapter": str(final_adapter_dir),
    }

    with open(
        run_dir / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()