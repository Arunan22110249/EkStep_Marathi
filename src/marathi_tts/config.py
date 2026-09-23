"""
Training configuration for Bodhan Marathi TTS fine-tuning.

Based on the Bodhan Indic-Speak token contract and inference
pipeline analysis.
"""

from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Bodhan model constants
# ============================================================================

BODHAN_VOCAB_SIZE = 156960

BODHAN_BOS_TOKEN_ID = 128000
BODHAN_EOS_TOKEN_ID = 128001

BODHAN_MODEL_MAX_LENGTH = 131072


# ============================================================================
# Bodhan control tokens
# ============================================================================

BODHAN_START_OF_HUMAN = 128259
BODHAN_END_OF_HUMAN = 128260

BODHAN_START_OF_AI = 128261
BODHAN_END_OF_AI = 128262

BODHAN_START_OF_SPEECH = 128257
BODHAN_END_OF_SPEECH = 128258


# ============================================================================
# Bodhan SNAC audio token contract
# ============================================================================

BODHAN_SNAC_START = 128266
BODHAN_SNAC_END = 156937

BODHAN_SNAC_COUNT = (
    BODHAN_SNAC_END
    - BODHAN_SNAC_START
    + 1
)

BODHAN_NUM_CODEBOOKS = 7
BODHAN_CODEBOOK_SIZE = 4096


# ============================================================================
# Training configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for Bodhan Marathi TTS fine-tuning."""

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    dataset_name: str = "SPRINGLab/IndicTTS_Marathi"
    dataset_split: str = "train"

    # None = use the complete training split.
    # Set to a small integer for smoke tests.
    max_samples: Optional[int] = None

    # ------------------------------------------------------------------------
    # Text and audio sequence limits
    # ------------------------------------------------------------------------

    # Maximum tokenizer length for the text/prompt portion.
    max_text_length: int = 128

    # Maximum number of Bodhan SNAC speech tokens.
    #
    # The actual per-example limit is determined by the remaining
    # max_seq_length budget after the prompt and special tokens.
    max_audio_tokens: int = 1536

    # Total model sequence length.
    #
    # Empirical profiling of 500 Marathi samples showed zero observed
    # truncation at 1536 tokens.
    max_seq_length: int = 1536

    # ------------------------------------------------------------------------
    # Speaker and style
    # ------------------------------------------------------------------------

    default_speaker: str = "Anagha"
    default_style: str = "NEUTRAL"

    # ------------------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------------------

    batch_size: int = 4

    learning_rate: float = 2e-5

    num_epochs: int = 1

    warmup_steps: int = 100

    # Gradient accumulation allows an effective larger batch size
    # when GPU memory is limited.
    gradient_accumulation_steps: int = 1

    # None = train for num_epochs.
    # An integer overrides epoch-based stopping and is useful for
    # smoke tests.
    max_steps: Optional[int] = None

    # ------------------------------------------------------------------------
    # LoRA configuration
    # ------------------------------------------------------------------------

    lora_r: int = 16

    lora_alpha: int = 32

    lora_dropout: float = 0.05

    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    # ------------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------------

    logging_steps: int = 1

    save_steps: int = 50

    output_dir: str = "outputs/checkpoints"

    # ------------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------------

    # Keep this at 0 for Windows/CPU compatibility.
    num_workers: int = 0

    prefetch_factor: int = 2

    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------

    def __post_init__(self):
        """Validate the training configuration."""

        # Model-level sequence limit.
        if self.max_seq_length > BODHAN_MODEL_MAX_LENGTH:
            raise ValueError(
                f"max_seq_length ({self.max_seq_length}) exceeds "
                f"BODHAN_MODEL_MAX_LENGTH ({BODHAN_MODEL_MAX_LENGTH})"
            )

        # Text length itself cannot exceed the model sequence limit.
        if self.max_text_length > self.max_seq_length:
            raise ValueError(
                f"max_text_length ({self.max_text_length}) exceeds "
                f"max_seq_length ({self.max_seq_length})"
            )

        # Audio token ceiling cannot exceed the model sequence limit.
        #
        # The actual training.py sequence builder performs the more
        # precise per-example calculation:
        #
        #   prompt_length
        #   + start_of_speech
        #   + speech_tokens
        #   + end_of_speech
        #
        # <= max_seq_length
        if self.max_audio_tokens > self.max_seq_length:
            raise ValueError(
                f"max_audio_tokens ({self.max_audio_tokens}) exceeds "
                f"max_seq_length ({self.max_seq_length})"
            )

        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {self.batch_size}"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )

        if self.num_epochs < 1:
            raise ValueError(
                f"num_epochs must be >= 1, got {self.num_epochs}"
            )

        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )

        if self.lora_r < 1:
            raise ValueError(
                f"lora_r must be >= 1, got {self.lora_r}"
            )

        if self.lora_alpha <= 0:
            raise ValueError(
                f"lora_alpha must be > 0, got {self.lora_alpha}"
            )

        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(
                "lora_dropout must be in [0, 1), "
                f"got {self.lora_dropout}"
            )

        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError(
                f"max_samples must be >= 1 or None, "
                f"got {self.max_samples}"
            )

        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(
                f"max_steps must be >= 1 or None, "
                f"got {self.max_steps}"
            )

        if self.logging_steps < 1:
            raise ValueError(
                f"logging_steps must be >= 1, "
                f"got {self.logging_steps}"
            )

        if self.save_steps < 1:
            raise ValueError(
                f"save_steps must be >= 1, "
                f"got {self.save_steps}"
            )