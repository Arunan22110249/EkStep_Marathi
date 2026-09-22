"""
Bodhan TTS Smoke Test — GPU-Ready Training Pipeline Validation.

Validates the complete training pipeline for Marathi TTS fine-tuning:
1. Environment detection (CPU/GPU)
2. Model and tokenizer loading
3. Dataset construction
4. Batch preparation
5. Forward pass
6. Loss computation
7. Backward pass
8. Optimizer step

Configuration:
- Configurable device (auto-detect: cuda/cpu)
- Configurable dtype (float32/float16/bfloat16)
- Configurable batch size
- Configurable gradient accumulation
- Configurable number of optimizer steps
- Configurable model path

Designed to run on GPU without model weight download blocker on CPU.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import traceback
from dataclasses import dataclass

import torch
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.marathi_tts.config import TrainingConfig
from src.marathi_tts.training import create_training_dataset


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SmokeTestConfig:
    """Configuration for Bodhan smoke test."""

    # Model configuration
    model_id: str = "bodhan-ai/indic-speak"

    # Environment configuration
    device: Optional[str] = None
    dtype: str = "float32"

    # Dataset configuration
    max_samples: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 1

    # Training configuration
    learning_rate: float = 1e-5
    num_steps: int = 5
    max_seq_length: Optional[int] = None

    # Optimization configuration
    use_gradient_checkpointing: bool = False
    low_cpu_mem_usage: bool = True

    # Reporting configuration
    log_memory_every_n_steps: int = 1

    def __post_init__(self):
        """Post-initialization validation and setup."""

        # Auto-detect device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if isinstance(self.dtype, str):
            self.dtype = dtype_map.get(self.dtype, torch.float32)


# Default configuration
DEFAULT_SMOKE_TEST_CONFIG = SmokeTestConfig()


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def detect_environment() -> Dict[str, Any]:
    """Detect and report on the environment."""

    env: Dict[str, Any] = {}

    # CUDA/GPU detection
    env["cuda_available"] = torch.cuda.is_available()
    env["torch_version"] = torch.__version__

    if torch.cuda.is_available():
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_names"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
        env["gpu_vram_gb"] = [
            torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in range(torch.cuda.device_count())
        ]
    else:
        env["gpu_count"] = 0
        env["gpu_names"] = []
        env["gpu_vram_gb"] = []

    # CPU detection
    try:
        import psutil

        env["cpu_count"] = psutil.cpu_count(logical=False)
        env["cpu_count_logical"] = psutil.cpu_count(logical=True)

        cpu_freq = psutil.cpu_freq()
        env["cpu_freq_ghz"] = (
            cpu_freq.current / 1000.0
            if cpu_freq
            else None
        )

        env["ram_gb"] = psutil.virtual_memory().total / 1e9
        env["ram_available_gb"] = psutil.virtual_memory().available / 1e9

    except ImportError:
        env["cpu_count"] = None
        env["cpu_count_logical"] = None
        env["cpu_freq_ghz"] = None
        env["ram_gb"] = None
        env["ram_available_gb"] = None

    return env


def report_environment(config: SmokeTestConfig):
    """Report environment and configuration."""

    env = detect_environment()

    logger.info("\n" + "=" * 80)
    logger.info("ENVIRONMENT DETECTION")
    logger.info("=" * 80)

    # PyTorch information
    logger.info(f"\nPyTorch Version: {env['torch_version']}")
    logger.info(f"Selected Device: {config.device.upper()}")
    logger.info(f"Data Type: {config.dtype}")

    # GPU information
    if env["cuda_available"]:
        logger.info("\n✓ CUDA Available")
        logger.info(f"  GPU Count: {env['gpu_count']}")

        for i, (name, vram) in enumerate(
            zip(env["gpu_names"], env["gpu_vram_gb"])
        ):
            logger.info(
                f"  GPU {i}: {name} ({vram:.2f} GB VRAM)"
            )
    else:
        logger.info("\n✗ CUDA Not Available (CPU-only mode)")

    # CPU information
    if env["cpu_count"] is not None:
        logger.info(
            f"\nCPU: {env['cpu_count']} cores "
            f"(logical: {env['cpu_count_logical']})"
        )

        if env["cpu_freq_ghz"]:
            logger.info(
                f"CPU Frequency: {env['cpu_freq_ghz']:.2f} GHz"
            )

        logger.info(
            f"Total RAM: {env['ram_gb']:.2f} GB"
        )
        logger.info(
            f"Available RAM: {env['ram_available_gb']:.2f} GB"
        )

    # Configuration summary
    logger.info("\nSmoke Test Configuration:")
    logger.info(f"  Model: {config.model_id}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Max Samples: {config.max_samples}")
    logger.info(f"  Num Steps: {config.num_steps}")
    logger.info(
        f"  Learning Rate: {config.learning_rate:.2e}"
    )
    logger.info(
        f"  Gradient Accumulation: "
        f"{config.gradient_accumulation_steps}"
    )


# ---------------------------------------------------------------------------
# Memory reporting
# ---------------------------------------------------------------------------

def log_memory_stats(
    step: Optional[int] = None,
    device: str = "cpu",
):
    """Log current memory usage."""

    if device == "cuda" and torch.cuda.is_available():

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9

        logger.info(
            f"  [Step {step}] GPU Memory: "
            f"{allocated:.2f}GB allocated, "
            f"{reserved:.2f}GB reserved"
        )

    elif device == "cpu":

        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()

            rss_mb = mem_info.rss / 1e6
            vms_mb = mem_info.vms / 1e6

            logger.info(
                f"  [Step {step}] CPU Memory: "
                f"{rss_mb:.1f}MB RSS, "
                f"{vms_mb:.1f}MB VMS"
            )

        except (ImportError, AttributeError):
            pass


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    config: SmokeTestConfig,
) -> Tuple[Any, Any]:
    """Load Bodhan model and tokenizer with memory optimizations."""

    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING MODEL & TOKENIZER")
    logger.info("=" * 80)

    # Tokenizer
    logger.info(
        f"\nLoading tokenizer from {config.model_id}..."
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id
        )

        logger.info("✓ Tokenizer loaded successfully")
        logger.info(
            f"  Vocab size: {len(tokenizer):,}"
        )

    except Exception as e:
        logger.error(
            f"✗ Failed to load tokenizer: {e}"
        )
        raise

    # Model
    logger.info(
        f"\nLoading model from {config.model_id}..."
    )
    logger.info(
        f"  Device: {config.device}"
    )
    logger.info(
        f"  Dtype: {config.dtype}"
    )
    logger.info(
        f"  Low CPU mem: {config.low_cpu_mem_usage}"
    )

    try:

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.dtype,
            device_map=config.device,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
        )

        logger.info(
            "✓ Model loaded successfully"
        )

        # Model statistics
        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        logger.info("\nModel Parameters:")
        logger.info(
            f"  Total: {total_params:,} "
            f"({total_params / 1e9:.2f}B)"
        )
        logger.info(
            f"  Trainable: {trainable_params:,} "
            f"({trainable_params / 1e9:.2f}B)"
        )

        log_memory_stats(
            step="after_load",
            device=config.device,
        )

        return tokenizer, model

    except Exception as e:

        logger.error(
            f"✗ Failed to load model: {e}"
        )

        if config.device == "cuda":
            logger.error(
                "  GPU may be out of memory "
                "or driver issue"
            )
        else:
            logger.error(
                "  CPU may not have enough memory "
                "(need ~30-50GB for Bodhan)"
            )

        logger.error(
            "  Model size: ~15GB (weights)"
        )

        raise


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

def create_training_config(
    config: SmokeTestConfig,
) -> TrainingConfig:
    """Create training config based on smoke test config."""

    train_config = TrainingConfig(
        max_samples=config.max_samples,
        batch_size=config.batch_size,
        num_epochs=1,
    )

    if config.max_seq_length is not None:
        train_config.max_seq_length = (
            config.max_seq_length
        )

    return train_config


# ---------------------------------------------------------------------------
# Batch validation / dtype normalization
# ---------------------------------------------------------------------------

def prepare_batch_for_model(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Move the batch to the target device and normalize
    model input tensor dtypes.

    Hugging Face causal language models expect:
      - input_ids: torch.long
      - labels: torch.long
      - attention_mask: integer/bool mask

    The dataset/collator can produce torch.int32 tensors.
    CUDA's NLLLoss/CrossEntropyLoss kernel does not support
    int32 labels, so labels must explicitly be converted
    to torch.long before the forward pass.
    """

    # Move tensors to device first.
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    # Token IDs must be int64.
    batch["input_ids"] = batch["input_ids"].long()

    # Labels MUST be int64 for CUDA CrossEntropy/NLL loss.
    batch["labels"] = batch["labels"].long()

    # Keep attention mask as integer-compatible tensor.
    batch["attention_mask"] = (
        batch["attention_mask"].long()
    )

    return batch


def validate_batch_dtypes(
    batch: Dict[str, torch.Tensor],
):
    """
    Validate that model inputs have CUDA-compatible dtypes.
    """

    if batch["input_ids"].dtype != torch.long:
        raise TypeError(
            "input_ids must be torch.long, "
            f"got {batch['input_ids'].dtype}"
        )

    if batch["labels"].dtype != torch.long:
        raise TypeError(
            "labels must be torch.long, "
            f"got {batch['labels'].dtype}"
        )

    if batch["attention_mask"].dtype != torch.long:
        raise TypeError(
            "attention_mask must be torch.long, "
            f"got {batch['attention_mask'].dtype}"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(
    config: SmokeTestConfig = None,
) -> bool:
    """Run the smoke test with specified configuration."""

    if config is None:
        config = DEFAULT_SMOKE_TEST_CONFIG

    logger.info("\n" + "=" * 80)
    logger.info(
        "BODHAN TTS SMOKE TEST — "
        "TRAINING PIPELINE VALIDATION"
    )
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Step 0: Environment
    # ------------------------------------------------------------------

    report_environment(config)

    # ------------------------------------------------------------------
    # Step 1: Model + tokenizer
    # ------------------------------------------------------------------

    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING MODEL & TOKENIZER")
    logger.info("=" * 80)

    try:

        tokenizer, model = (
            load_model_and_tokenizer(config)
        )

        log_memory_stats(
            step="after_load",
            device=config.device,
        )

    except Exception as e:

        logger.error(
            "\n✗ SMOKE TEST FAILED at model loading"
        )
        logger.error(
            f"  Error: {e}"
        )

        if config.device == "cpu":
            logger.error(
                "\nThis is expected on CPU-only machines. "
                "Use a GPU instance for training."
            )

        return False

    # ------------------------------------------------------------------
    # Step 2: Dataset
    # ------------------------------------------------------------------

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CREATING DATASET")
    logger.info("=" * 80)

    try:

        train_config = create_training_config(
            config
        )

        logger.info("Training config:")
        logger.info(
            f"  Dataset: "
            f"{train_config.dataset_name}"
        )
        logger.info(
            f"  Max samples: "
            f"{train_config.max_samples}"
        )
        logger.info(
            f"  Batch size: "
            f"{config.batch_size}"
        )
        logger.info(
            f"  Max seq length: "
            f"{train_config.max_seq_length}"
        )

        logger.info(
            "\nLoading dataset..."
        )

        dataset, collator, cfg = (
            create_training_dataset(
                train_config,
                tokenizer,
            )
        )

        logger.info(
            f"✓ Dataset created with "
            f"{len(dataset)} samples"
        )

        log_memory_stats(
            step="after_dataset_load",
            device=config.device,
        )

    except Exception as e:

        logger.error(
            "\n✗ SMOKE TEST FAILED at dataset creation"
        )
        logger.error(
            f"  Error: {e}\n"
            f"{traceback.format_exc()}"
        )

        return False

    # ------------------------------------------------------------------
    # Step 3: Optimizer
    # ------------------------------------------------------------------

    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CREATING OPTIMIZER")
    logger.info("=" * 80)

    try:

        optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
        )

        logger.info(
            "✓ Optimizer created "
            f"(Adam, lr={config.learning_rate:.2e})"
        )

        log_memory_stats(
            step="after_optimizer",
            device=config.device,
        )

    except Exception as e:

        logger.error(
            "\n✗ SMOKE TEST FAILED at optimizer creation"
        )
        logger.error(
            f"  Error: {e}"
        )

        return False

    # ------------------------------------------------------------------
    # Step 4: Forward / backward / optimizer
    # ------------------------------------------------------------------

    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: RUNNING TRAINING STEPS")
    logger.info("=" * 80)

    logger.info(
        f"Will run {config.num_steps} "
        "optimizer steps...\n"
    )

    model.train()

    step_results = []

    forward_pass_successful = False
    backward_pass_successful = False
    optimizer_step_successful = False

    try:

        for step in range(config.num_steps):

            # ----------------------------------------------------------
            # Get batch
            # ----------------------------------------------------------

            batch_indices = list(
                range(
                    min(
                        config.batch_size,
                        len(dataset),
                    )
                )
            )

            batch_data = [
                dataset[i]
                for i in batch_indices
            ]

            batch = collator(batch_data)

            # ----------------------------------------------------------
            # Report original batch structure
            # ----------------------------------------------------------

            if step == 0:

                logger.info(
                    "Batch Structure (Step 1):"
                )

                logger.info(
                    f"  input_ids: "
                    f"{batch['input_ids'].shape} "
                    f"(dtype: "
                    f"{batch['input_ids'].dtype})"
                )

                logger.info(
                    f"  labels: "
                    f"{batch['labels'].shape} "
                    f"(dtype: "
                    f"{batch['labels'].dtype})"
                )

                logger.info(
                    f"  attention_mask: "
                    f"{batch['attention_mask'].shape} "
                    f"(dtype: "
                    f"{batch['attention_mask'].dtype})"
                )

            # ----------------------------------------------------------
            # Move to device + FIX INTEGER DTYPES
            # ----------------------------------------------------------

            batch = prepare_batch_for_model(
                batch,
                config.device,
            )

            # ----------------------------------------------------------
            # Validate corrected dtypes
            # ----------------------------------------------------------

            validate_batch_dtypes(batch)

            if step == 0:

                logger.info(
                    "\nCorrected Training Tensor Dtypes:"
                )

                logger.info(
                    f"  input_ids: "
                    f"{batch['input_ids'].dtype}"
                )

                logger.info(
                    f"  labels: "
                    f"{batch['labels'].dtype}"
                )

                logger.info(
                    f"  attention_mask: "
                    f"{batch['attention_mask'].dtype}"
                )

                logger.info(
                    "  ✓ All training tensors "
                    "normalized to torch.long"
                )

            # ----------------------------------------------------------
            # Forward pass
            # ----------------------------------------------------------

            outputs = model(**batch)

            loss = outputs.loss

            forward_pass_successful = True

            logger.info(
                f"  Forward pass successful "
                f"(loss={loss.item():.6f})"
            )

            # ----------------------------------------------------------
            # Backward pass
            # ----------------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            backward_pass_successful = True

            logger.info(
                "  Backward pass successful"
            )

            # ----------------------------------------------------------
            # Optimizer step
            # ----------------------------------------------------------

            optimizer.step()

            optimizer_step_successful = True

            logger.info(
                "  Optimizer step successful"
            )

            # ----------------------------------------------------------
            # Log results
            # ----------------------------------------------------------

            loss_val = loss.item()

            step_results.append(
                {
                    "step": step + 1,
                    "loss": loss_val,
                    "batch_size": batch[
                        "input_ids"
                    ].shape[0],
                    "seq_length": batch[
                        "input_ids"
                    ].shape[1],
                }
            )

            logger.info(
                f"  Step {step + 1}/"
                f"{config.num_steps}: "
                f"loss={loss_val:.6f}, "
                f"batch="
                f"{batch['input_ids'].shape}, "
                f"device={config.device}"
            )

            if (
                (step + 1)
                % config.log_memory_every_n_steps
                == 0
            ):
                log_memory_stats(
                    step=step + 1,
                    device=config.device,
                )

        # --------------------------------------------------------------
        # Training completed
        # --------------------------------------------------------------

        logger.info(
            f"\n✓ All {config.num_steps} "
            "training steps completed successfully"
        )

        log_memory_stats(
            step="final",
            device=config.device,
        )

    except RuntimeError as e:

        logger.error(
            "\n✗ SMOKE TEST FAILED "
            "during training step"
        )

        logger.error(
            f"  Error type: {type(e).__name__}"
        )

        logger.error(
            f"  Error message: {e}"
        )

        if "out of memory" in str(e).lower():

            if config.device == "cuda":

                logger.error(
                    "\n  → GPU out of memory"
                )

                logger.error(
                    "     Try: smaller batch_size, "
                    "shorter max_seq_length, "
                    "or reduce num_steps"
                )

            else:

                logger.error(
                    "\n  → CPU out of memory"
                )

                logger.error(
                    "     Use a GPU instance instead"
                )

        logger.error(
            "\nFull traceback:"
        )

        logger.error(
            traceback.format_exc()
        )

        return False

    except Exception as e:

        logger.error(
            "\n✗ SMOKE TEST FAILED "
            "with unexpected error"
        )

        logger.error(
            f"  Error: {e}\n"
            f"{traceback.format_exc()}"
        )

        return False

    # ------------------------------------------------------------------
    # Step 5: Results
    # ------------------------------------------------------------------

    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: RESULTS & SUMMARY")
    logger.info("=" * 80)

    logger.info("\nTraining Metrics:")

    logger.info(
        "  Forward pass: "
        f"{'✓ Success' if forward_pass_successful else '✗ Failed'}"
    )

    logger.info(
        "  Backward pass: "
        f"{'✓ Success' if backward_pass_successful else '✗ Failed'}"
    )

    logger.info(
        "  Optimizer step: "
        f"{'✓ Success' if optimizer_step_successful else '✗ Failed'}"
    )

    logger.info(
        f"\nLoss Progression "
        f"({config.num_steps} steps):"
    )

    for result in step_results:

        logger.info(
            f"  Step {result['step']:2d}: "
            f"loss={result['loss']:.6f} "
            f"(batch={result['batch_size']}, "
            f"seq_len={result['seq_length']})"
        )

    # ------------------------------------------------------------------
    # Loss trend
    # ------------------------------------------------------------------

    losses = [
        result["loss"]
        for result in step_results
    ]

    logger.info(
        "\nLoss Trend Analysis:"
    )

    logger.info(
        f"  Initial loss: "
        f"{losses[0]:.6f}"
    )

    logger.info(
        f"  Final loss: "
        f"{losses[-1]:.6f}"
    )

    loss_change = (
        losses[-1] - losses[0]
    )

    loss_change_pct = (
        loss_change / losses[0]
    ) * 100

    logger.info(
        f"  Change: "
        f"{loss_change:.6f} "
        f"({loss_change_pct:+.1f}%)"
    )

    if losses[-1] < losses[0]:

        logger.info(
            "  ✓ Loss decreasing "
            "(expected for training)"
        )

    else:

        logger.info(
            "  ⚠ Loss not decreasing "
            "(may indicate learning rate "
            "or data issue)"
        )

    # ------------------------------------------------------------------
    # Final success
    # ------------------------------------------------------------------

    logger.info("\n" + "=" * 80)
    logger.info("✓ SMOKE TEST PASSED")
    logger.info("=" * 80)

    logger.info(
        "\nValidated Pipeline Components:"
    )

    logger.info(
        "  ✓ Environment detection (GPU/CPU)"
    )
    logger.info(
        "  ✓ Model loading"
    )
    logger.info(
        "  ✓ Dataset construction"
    )
    logger.info(
        "  ✓ Batch preparation"
    )
    logger.info(
        "  ✓ Batch shapes and dtypes"
    )
    logger.info(
        "  ✓ CUDA-compatible training dtypes"
    )
    logger.info(
        "  ✓ Forward pass & loss computation"
    )
    logger.info(
        "  ✓ Backward pass & gradients"
    )
    logger.info(
        "  ✓ Optimizer step"
    )

    logger.info(
        f"\nCompleted {config.num_steps} "
        "training steps successfully"
    )

    logger.info(
        f"Model: {config.model_id}"
    )

    logger.info(
        f"Device: {config.device}"
    )

    logger.info(
        "Ready to proceed with full training"
    )

    logger.info("=" * 80)

    return True


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Bodhan TTS Smoke Test — "
            "GPU-Ready Training Pipeline Validation"
        )
    )

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------

    parser.add_argument(
        "--model-id",
        type=str,
        default="bodhan-ai/indic-speak",
        help="Hugging Face model ID to load",
    )

    # ------------------------------------------------------------------
    # Device configuration
    # ------------------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Device to use "
            "(auto=detect, cpu=force CPU, "
            "cuda=force GPU)"
        ),
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=[
            "float32",
            "float16",
            "bfloat16",
        ],
        default="float32",
        help="Data type for model weights",
    )

    # ------------------------------------------------------------------
    # Dataset configuration
    # ------------------------------------------------------------------

    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help=(
            "Maximum number of samples "
            "to use from dataset"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per step",
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=(
            "Maximum sequence length "
            "(None=use default 2600)"
        ),
    )

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for optimizer",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of optimizer steps to run",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # ------------------------------------------------------------------
    # Memory configuration
    # ------------------------------------------------------------------

    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Enable gradient checkpointing "
            "(saves memory)"
        ),
    )

    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        default=True,
        help=(
            "Use low CPU memory loading strategy"
        ),
    )

    # ------------------------------------------------------------------
    # Logging configuration
    # ------------------------------------------------------------------

    parser.add_argument(
        "--log-memory-every-n-steps",
        type=int,
        default=1,
        help="Log memory every N steps",
    )

    # ------------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------------

    args = parser.parse_args()

    # Auto device → None
    device = (
        args.device
        if args.device != "auto"
        else None
    )

    # Create configuration
    config = SmokeTestConfig(
        model_id=args.model_id,
        device=device,
        dtype=args.dtype,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        max_seq_length=args.max_seq_length,
        use_gradient_checkpointing=(
            args.gradient_checkpointing
        ),
        low_cpu_mem_usage=(
            args.low_cpu_mem_usage
        ),
        log_memory_every_n_steps=(
            args.log_memory_every_n_steps
        ),
    )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    try:

        success = run_smoke_test(config)

        sys.exit(
            0 if success else 1
        )

    except KeyboardInterrupt:

        logger.warning(
            "\nSmoke test interrupted by user"
        )

        sys.exit(1)

    except Exception as e:

        logger.error(
            f"\nUnexpected error in smoke test: {e}"
        )

        logger.error(
            traceback.format_exc()
        )

        sys.exit(1)