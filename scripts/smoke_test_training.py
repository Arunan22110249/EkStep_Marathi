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
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.marathi_tts.config import TrainingConfig
from src.marathi_tts.training import create_training_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SmokeTestConfig:
    """Configuration for Bodhan smoke test."""
    
    # Model config
    model_id: str = "bodhan-ai/indic-speak"
    
    # Environment config
    device: Optional[str] = None  # None = auto-detect (cuda if available, else cpu)
    dtype: str = "float32"  # "float32", "float16", "bfloat16"
    
    # Dataset config
    max_samples: int = 10  # Number of samples from Marathi dataset
    batch_size: int = 1  # Batch size per step
    gradient_accumulation_steps: int = 1  # Gradient accumulation
    
    # Training config
    learning_rate: float = 1e-5
    num_steps: int = 5  # Number of optimizer steps to run
    max_seq_length: Optional[int] = None  # None = use default (2600)
    
    # Optimization config
    use_gradient_checkpointing: bool = False
    low_cpu_mem_usage: bool = True
    
    # Reporting config
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


# Default smoke test configuration
DEFAULT_SMOKE_TEST_CONFIG = SmokeTestConfig()


def detect_environment() -> Dict[str, Any]:
    """Detect and report on the environment."""
    env = {}
    
    # CUDA/GPU detection
    env["cuda_available"] = torch.cuda.is_available()
    env["torch_version"] = torch.__version__
    
    if torch.cuda.is_available():
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        env["gpu_vram_gb"] = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(torch.cuda.device_count())]
    else:
        env["gpu_count"] = 0
        env["gpu_names"] = []
        env["gpu_vram_gb"] = []
    
    # CPU detection
    try:
        import psutil
        env["cpu_count"] = psutil.cpu_count(logical=False)
        env["cpu_count_logical"] = psutil.cpu_count(logical=True)
        env["cpu_freq_ghz"] = psutil.cpu_freq().current / 1000.0 if psutil.cpu_freq() else None
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
    
    # PyTorch info
    logger.info(f"\nPyTorch Version: {env['torch_version']}")
    logger.info(f"Selected Device: {config.device.upper()}")
    logger.info(f"Data Type: {config.dtype}")
    
    # GPU info
    if env["cuda_available"]:
        logger.info(f"\n✓ CUDA Available")
        logger.info(f"  GPU Count: {env['gpu_count']}")
        for i, (name, vram) in enumerate(zip(env['gpu_names'], env['gpu_vram_gb'])):
            logger.info(f"  GPU {i}: {name} ({vram:.2f} GB VRAM)")
    else:
        logger.info(f"\n✗ CUDA Not Available (CPU-only mode)")
    
    # CPU info
    if env["cpu_count"] is not None:
        logger.info(f"\nCPU: {env['cpu_count']} cores (logical: {env['cpu_count_logical']})")
        if env["cpu_freq_ghz"]:
            logger.info(f"CPU Frequency: {env['cpu_freq_ghz']:.2f} GHz")
        logger.info(f"Total RAM: {env['ram_gb']:.2f} GB")
        logger.info(f"Available RAM: {env['ram_available_gb']:.2f} GB")
    
    # Config summary
    logger.info(f"\nSmoke Test Configuration:")
    logger.info(f"  Model: {config.model_id}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Max Samples: {config.max_samples}")
    logger.info(f"  Num Steps: {config.num_steps}")
    logger.info(f"  Learning Rate: {config.learning_rate:.2e}")
    logger.info(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")


def log_memory_stats(step: Optional[int] = None, device: str = "cpu"):
    """Log current memory usage."""
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            f"  [Step {step}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
    elif device == "cpu":
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / 1e6
            vms_mb = mem_info.vms / 1e6
            logger.info(f"  [Step {step}] CPU Memory: {rss_mb:.1f}MB RSS, {vms_mb:.1f}MB VMS")
        except (ImportError, AttributeError):
            pass


def load_model_and_tokenizer(config: SmokeTestConfig) -> Tuple[Any, Any]:
    """Load Bodhan model and tokenizer with memory optimizations."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING MODEL & TOKENIZER")
    logger.info("=" * 80)
    
    logger.info(f"\nLoading tokenizer from {config.model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        logger.info(f"✓ Tokenizer loaded successfully")
        logger.info(f"  Vocab size: {len(tokenizer):,}")
    except Exception as e:
        logger.error(f"✗ Failed to load tokenizer: {e}")
        raise

    logger.info(f"\nLoading model from {config.model_id}...")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Dtype: {config.dtype}")
    logger.info(f"  Low CPU mem: {config.low_cpu_mem_usage}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.dtype,
            device_map=config.device,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
        )
        logger.info(f"✓ Model loaded successfully")
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"\nModel Parameters:")
        logger.info(f"  Total: {total_params:,} ({total_params/1e9:.2f}B)")
        logger.info(f"  Trainable: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
        
        log_memory_stats(step="after_load", device=config.device)
        return tokenizer, model
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        if config.device == "cuda":
            logger.error(f"  GPU may be out of memory or driver issue")
        else:
            logger.error(f"  CPU may not have enough memory (need ~30-50GB for Bodhan)")
        logger.error(f"  Model size: ~15GB (weights)")
        raise


def create_training_config(config: SmokeTestConfig) -> TrainingConfig:
    """Create training config based on smoke test config."""
    train_config = TrainingConfig(
        max_samples=config.max_samples,
        batch_size=config.batch_size,
        num_epochs=1,
    )
    if config.max_seq_length is not None:
        train_config.max_seq_length = config.max_seq_length
    
    return train_config


def run_smoke_test(config: SmokeTestConfig = None) -> bool:
    """Run the smoke test with specified configuration."""
    if config is None:
        config = DEFAULT_SMOKE_TEST_CONFIG
    
    logger.info("\n" + "=" * 80)
    logger.info("BODHAN TTS SMOKE TEST — TRAINING PIPELINE VALIDATION")
    logger.info("=" * 80)
    
    # Step 0: Environment detection
    report_environment(config)
    
    # Step 1: Load model and tokenizer
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING MODEL & TOKENIZER")
    logger.info("=" * 80)
    try:
        tokenizer, model = load_model_and_tokenizer(config)
        log_memory_stats(step="after_load", device=config.device)
    except Exception as e:
        logger.error(f"\n✗ SMOKE TEST FAILED at model loading")
        logger.error(f"  Error: {e}")
        if config.device == "cpu":
            logger.error(f"\nThis is expected on CPU-only machines. Use a GPU instance for training.")
        return False
    
    # Step 2: Create config and dataset
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CREATING DATASET")
    logger.info("=" * 80)
    try:
        train_config = create_training_config(config)
        logger.info(f"Training config:")
        logger.info(f"  Dataset: {train_config.dataset_name}")
        logger.info(f"  Max samples: {train_config.max_samples}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Max seq length: {train_config.max_seq_length}")
        
        logger.info(f"\nLoading dataset...")
        dataset, collator, cfg = create_training_dataset(train_config, tokenizer)
        logger.info(f"✓ Dataset created with {len(dataset)} samples")
        log_memory_stats(step="after_dataset_load", device=config.device)
    except Exception as e:
        logger.error(f"\n✗ SMOKE TEST FAILED at dataset creation")
        logger.error(f"  Error: {e}\n{traceback.format_exc()}")
        return False
    
    # Step 3: Create optimizer
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: CREATING OPTIMIZER")
    logger.info("=" * 80)
    try:
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        logger.info(f"✓ Optimizer created (Adam, lr={config.learning_rate:.2e})")
        log_memory_stats(step="after_optimizer", device=config.device)
    except Exception as e:
        logger.error(f"\n✗ SMOKE TEST FAILED at optimizer creation")
        logger.error(f"  Error: {e}")
        return False
    
    # Step 4: Run forward/backward passes
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: RUNNING TRAINING STEPS")
    logger.info("=" * 80)
    logger.info(f"Will run {config.num_steps} optimizer steps...\n")
    
    model.train()
    step_results = []
    forward_pass_successful = False
    backward_pass_successful = False
    optimizer_step_successful = False
    
    try:
        for step in range(config.num_steps):
            # Get a batch
            batch_indices = list(range(min(config.batch_size, len(dataset))))
            batch_data = [dataset[i] for i in batch_indices]
            batch = collator(batch_data)
            
            # Report batch structure (first step only)
            if step == 0:
                logger.info(f"Batch Structure (Step 1):")
                logger.info(f"  input_ids: {batch['input_ids'].shape} (dtype: {batch['input_ids'].dtype})")
                logger.info(f"  labels: {batch['labels'].shape} (dtype: {batch['labels'].dtype})")
                logger.info(f"  attention_mask: {batch['attention_mask'].shape} (dtype: {batch['attention_mask'].dtype})")
            
            # Move to device
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            forward_pass_successful = True
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            backward_pass_successful = True
            
            # Optimizer step
            optimizer.step()
            optimizer_step_successful = True
            
            # Log results
            loss_val = loss.item()
            step_results.append({
                "step": step + 1,
                "loss": loss_val,
                "batch_size": batch["input_ids"].shape[0],
                "seq_length": batch["input_ids"].shape[1],
            })
            
            logger.info(
                f"  Step {step + 1}/{config.num_steps}: "
                f"loss={loss_val:.6f}, batch={batch['input_ids'].shape}, "
                f"device={config.device}"
            )
            
            if (step + 1) % config.log_memory_every_n_steps == 0:
                log_memory_stats(step=step + 1, device=config.device)
        
        logger.info(f"\n✓ All {config.num_steps} training steps completed successfully")
        log_memory_stats(step="final", device=config.device)
        
    except RuntimeError as e:
        logger.error(f"\n✗ SMOKE TEST FAILED during training step")
        logger.error(f"  Error type: {type(e).__name__}")
        logger.error(f"  Error message: {e}")
        if "out of memory" in str(e).lower():
            if config.device == "cuda":
                logger.error(f"\n  → GPU out of memory")
                logger.error(f"     Try: smaller batch_size, shorter max_seq_length, or reduce num_steps")
            else:
                logger.error(f"\n  → CPU out of memory")
                logger.error(f"     Use a GPU instance instead")
        logger.error(f"\nFull traceback:")
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"\n✗ SMOKE TEST FAILED with unexpected error")
        logger.error(f"  Error: {e}\n{traceback.format_exc()}")
        return False
    
    # Step 5: Report results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: RESULTS & SUMMARY")
    logger.info("=" * 80)
    
    logger.info("\nTraining Metrics:")
    logger.info(f"  Forward pass: {'✓ Success' if forward_pass_successful else '✗ Failed'}")
    logger.info(f"  Backward pass: {'✓ Success' if backward_pass_successful else '✗ Failed'}")
    logger.info(f"  Optimizer step: {'✓ Success' if optimizer_step_successful else '✗ Failed'}")
    
    logger.info(f"\nLoss Progression ({config.num_steps} steps):")
    for result in step_results:
        logger.info(
            f"  Step {result['step']:2d}: loss={result['loss']:.6f} "
            f"(batch={result['batch_size']}, seq_len={result['seq_length']})"
        )
    
    # Compute loss trend
    losses = [r['loss'] for r in step_results]
    logger.info(f"\nLoss Trend Analysis:")
    logger.info(f"  Initial loss: {losses[0]:.6f}")
    logger.info(f"  Final loss: {losses[-1]:.6f}")
    loss_change = losses[-1] - losses[0]
    loss_change_pct = (loss_change / losses[0]) * 100
    logger.info(f"  Change: {loss_change:.6f} ({loss_change_pct:+.1f}%)")
    if losses[-1] < losses[0]:
        logger.info(f"  ✓ Loss decreasing (expected for training)")
    else:
        logger.info(f"  ⚠ Loss not decreasing (may indicate learning rate or data issue)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ SMOKE TEST PASSED")
    logger.info("=" * 80)
    logger.info("\nValidated Pipeline Components:")
    logger.info("  ✓ Environment detection (GPU/CPU)")
    logger.info("  ✓ Model loading")
    logger.info("  ✓ Dataset construction")
    logger.info("  ✓ Batch preparation")
    logger.info("  ✓ Batch shapes and dtypes")
    logger.info("  ✓ Forward pass & loss computation")
    logger.info("  ✓ Backward pass & gradients")
    logger.info("  ✓ Optimizer step")
    logger.info(f"\nCompleted {config.num_steps} training steps successfully")
    logger.info(f"Model: {config.model_id}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Ready to proceed with full training")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bodhan TTS Smoke Test — GPU-Ready Training Pipeline Validation"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-id",
        type=str,
        default="bodhan-ai/indic-speak",
        help="Hugging Face model ID to load"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use (auto=detect, cpu=force CPU, cuda=force GPU)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Data type for model weights"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of samples to use from dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per step"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (None=use default 2600)"
    )
    
    # Training configuration
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of optimizer steps to run"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    
    # Memory configuration
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (saves memory)"
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        default=True,
        help="Use low CPU memory loading strategy"
    )
    
    parser.add_argument(
        "--log-memory-every-n-steps",
        type=int,
        default=1,
        help="Log memory every N steps"
    )
    
    args = parser.parse_args()
    
    # Create config from arguments
    device = args.device if args.device != "auto" else None  # None = auto-detect
    config = SmokeTestConfig(
        model_id=args.model_id,
        device=device,
        dtype=args.dtype,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        max_seq_length=args.max_seq_length,
        use_gradient_checkpointing=args.gradient_checkpointing,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        log_memory_every_n_steps=args.log_memory_every_n_steps,
    )
    
    try:
        success = run_smoke_test(config)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\nSmoke test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error in smoke test: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
