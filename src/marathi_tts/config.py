"""
Training configuration for Bodhan Marathi TTS fine-tuning.

Based on Bodhan Indic-Speak token contract and inference pipeline analysis.
"""

from dataclasses import dataclass
from typing import Optional


# Bodhan model constants (from authenticated inspection)
BODHAN_VOCAB_SIZE = 156960
BODHAN_BOS_TOKEN_ID = 128000
BODHAN_EOS_TOKEN_ID = 128001
BODHAN_MODEL_MAX_LENGTH = 131072

# Control tokens (from token_contract.md)
BODHAN_START_OF_HUMAN = 128259
BODHAN_END_OF_HUMAN = 128260
BODHAN_START_OF_AI = 128261
BODHAN_END_OF_AI = 128262
BODHAN_START_OF_SPEECH = 128257
BODHAN_END_OF_SPEECH = 128258

# SNAC audio code range (from token_contract.md)
BODHAN_SNAC_START = 128266
BODHAN_SNAC_END = 156937
BODHAN_SNAC_COUNT = BODHAN_SNAC_END - BODHAN_SNAC_START + 1
BODHAN_NUM_CODEBOOKS = 7
BODHAN_CODEBOOK_SIZE = 4096


@dataclass
class TrainingConfig:
    """Configuration for Bodhan fine-tuning."""
    
    # Dataset config
    dataset_name: str = "SPRINGLab/IndicTTS_Marathi"
    dataset_split: str = "train"
    max_samples: Optional[int] = None  # None = use all; for testing use 10
    
    # Text and audio processing
    max_text_length: int = 512  # Max tokens for text (before speech)
    max_audio_tokens: int = 2048  # Max SNAC tokens per sample (~8 seconds at 24 kHz)
    max_seq_length: int = 2600  # Total sequence length (text + speech + special tokens + buffer)
    
    # Speaker and style (from inference.py)
    default_speaker: str = "Anagha"  # Female Marathi voice from voices.md
    default_style: str = "NEUTRAL"
    
    # Training hyperparameters (for later use)
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 1
    warmup_steps: int = 100
    
    # Data loading
    num_workers: int = 0  # CPU-only environment
    prefetch_factor: int = 2
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_seq_length > BODHAN_MODEL_MAX_LENGTH:
            raise ValueError(
                f"max_seq_length ({self.max_seq_length}) exceeds "
                f"BODHAN_MODEL_MAX_LENGTH ({BODHAN_MODEL_MAX_LENGTH})"
            )
        
        if self.max_text_length + self.max_audio_tokens > self.max_seq_length - 20:
            # Reserve ~20 tokens for special tokens and safety margin
            raise ValueError(
                f"max_text_length ({self.max_text_length}) + "
                f"max_audio_tokens ({self.max_audio_tokens}) exceeds "
                f"max_seq_length ({self.max_seq_length})"
            )
