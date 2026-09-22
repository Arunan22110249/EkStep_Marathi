"""
Training dataset for Bodhan fine-tuning on Marathi TTS.

Handles:
- Loading Marathi audio + text pairs
- Building Bodhan prompts with speaker/style
- Tokenizing text and speech tokens
- Creating causal LM labels (supervise only speech tokens)
- Batch collation with padding and attention masks
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
import torch
from pathlib import Path

from .config import (
    TrainingConfig,
    BODHAN_START_OF_HUMAN,
    BODHAN_END_OF_HUMAN,
    BODHAN_START_OF_AI,
    BODHAN_END_OF_AI,
    BODHAN_START_OF_SPEECH,
    BODHAN_END_OF_SPEECH,
    BODHAN_SNAC_START,
    BODHAN_NUM_CODEBOOKS,
)
from .preprocess import DataPipelineProcessor

logger = logging.getLogger(__name__)


class BodhanPromptBuilder:
    """Build Bodhan prompt strings following the inference.py template."""
    
    @staticmethod
    def build_prompt(
        text: str,
        speaker: str = "Anagha",
        style: str = "NEUTRAL",
    ) -> str:
        """
        Build Bodhan prompt string.
        
        Format:
        <|start_of_human|><|begin_of_text|>
        [<|speaker>Speaker Name<speaker|>]
        [<|style>STYLE<style|>]
        {text}
        <|eot_id|>
        <|end_of_human|>
        <|start_of_ai|>
        <|start_of_speech|>
        
        Note: <|start_of_speech|> is the first GENERATED token, not part of the prompt context.
        
        Args:
            text: Marathi text to synthesize
            speaker: Speaker name (e.g., "Anagha", "Chinmay")
            style: Emotion/style (e.g., "NEUTRAL", "HAPPY")
        
        Returns:
            Prompt string without speech tokens
        """
        # Build blocks following inference.py structure
        prompt_parts = [
            "<|start_of_human|>",
            "<|begin_of_text|>",
        ]
        
        # Add speaker block if provided
        if speaker:
            prompt_parts.append(f"<|speaker>{speaker}<speaker|>")
        
        # Add style block if provided
        if style:
            prompt_parts.append(f"<|style>{style}<style|>")
        
        # Add text and closing
        # Note: <|start_of_speech|> is NOT in the prompt context.
        # It is the FIRST GENERATED token in the audio span (from inference.py).
        # It will be added explicitly during training dataset construction.
        prompt_parts.extend([
            text,
            "<|eot_id|>",
            "<|end_of_human|>",
            "<|start_of_ai|>",
        ])
        
        return "".join(prompt_parts)


class BodhanTrainingDataset:
    """
    Dataset for training Bodhan on Marathi TTS.
    
    Input: Text + Marathi audio at 48 kHz
    Process: Text validation → Audio resample (48→24 kHz) → SNAC encode → Bodhan tokenize
    Output: (input_ids, labels, attention_mask) for causal LM training
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        tokenizer: Any = None,
        preprocessing_module: Optional[DataPipelineProcessor] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            config: TrainingConfig with dataset and sequence settings
            tokenizer: HF tokenizer for Bodhan vocab (loads if None)
            preprocessing_module: DataPipelineProcessor for audio/text validation
        """
        self.config = config
        self.tokenizer = tokenizer or self._load_tokenizer()
        self.preprocessor = preprocessing_module or DataPipelineProcessor(device="cpu")
        self.prompt_builder = BodhanPromptBuilder()
        self.dataset = None
        self._load_dataset()
    
    def _load_tokenizer(self):
        """Load Bodhan tokenizer from Hugging Face."""
        logger.info("Loading Bodhan tokenizer from bodhan-ai/indic-speak...")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "bodhan-ai/indic-speak",
                trust_remote_code=True,
            )
            logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load Bodhan tokenizer: {e}")
            logger.info("Using placeholder tokenizer for validation mode...")
            return None
    
    def _load_dataset(self):
        """Load Marathi TTS dataset."""
        logger.info(f"Loading {self.config.dataset_name} ({self.config.dataset_split})...")
        try:
            from datasets import load_dataset
            
            ds = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split,
                streaming=False,
            )
            
            # Use raw format to avoid torchcodec dependency
            ds = ds.with_format("arrow")
            
            # Limit to subset if configured
            if self.config.max_samples:
                ds = ds.select(range(min(self.config.max_samples, len(ds))))
            
            self.dataset = ds
            logger.info(f"Loaded {len(self.dataset)} examples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset) if self.dataset else 0
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get one training example.
        
        Returns:
            Dict with:
            - input_ids: Token IDs (text prompt + speech tokens)
            - labels: Causal LM labels (-100 for text, token IDs for speech)
            - attention_mask: Attention mask (1 for valid, 0 for padding)
            - text_length: Number of tokens in text portion
            - speech_length: Number of SNAC tokens
        
        Raises:
            ValueError: If processing fails
        """
        # Load example
        example = self.dataset[idx]
        
        # Convert PyArrow table to dict if needed
        if hasattr(example, 'to_pydict'):
            example = example.to_pydict()
        
        # Process through pipeline (validates text, resamples audio, encodes SNAC)
        try:
            result = self.preprocessor.process(example)
        except Exception as e:
            raise ValueError(f"Preprocessing failed for example {idx}: {e}")
        
        # Extract components
        text = result["text"]
        speech_tokens = result["tokens"]  # Bodhan-serialized token IDs
        
        # Build prompt (without speech tokens yet)
        prompt_str = self.prompt_builder.build_prompt(
            text=text,
            speaker=self.config.default_speaker,
            style=self.config.default_style,
        )
        
        # Tokenize prompt
        if self.tokenizer is None:
            # Placeholder mode for testing: use mock tokenization
            prompt_token_ids = list(range(10, 10 + len(prompt_str.split())))
        else:
            encoding = self.tokenizer(
                prompt_str,
                add_special_tokens=False,
                return_tensors=None,
                truncation=True,
                max_length=self.config.max_text_length,
            )
            prompt_token_ids = encoding["input_ids"]
        
        text_length = len(prompt_token_ids)
        
        # Ensure speech tokens fit in sequence
        max_speech_tokens = self.config.max_seq_length - text_length - 10  # 10-token safety margin
        if len(speech_tokens) > max_speech_tokens:
            logger.warning(
                f"Example {idx}: {len(speech_tokens)} speech tokens exceed "
                f"max {max_speech_tokens}. Truncating."
            )
            speech_tokens = speech_tokens[:max_speech_tokens]
        
        # Construct full input: prompt + start_of_speech + speech_codes + end_of_speech
        input_ids = (
            prompt_token_ids +
            [BODHAN_START_OF_SPEECH] +
            speech_tokens.tolist() +
            [BODHAN_END_OF_SPEECH]
        )
        
        # Create labels: -100 (ignore) for prompt only, supervise speech tokens.
        # The model must learn to generate start_of_speech after seeing <|start_of_ai|>,
        # so it is supervised (not masked) in the labels.
        labels = (
            [-100] * text_length +         # Only text tokens are masked (ignored in loss)
            [BODHAN_START_OF_SPEECH] +     # start_of_speech is supervised (model learns to generate this)
            speech_tokens.tolist() +       # Speech tokens are supervised
            [BODHAN_END_OF_SPEECH]         # End token is supervised
        )
        
        # Verify lengths match
        assert len(input_ids) == len(labels), "Input and label lengths mismatch"
        
        return {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32),
            "text_length": text_length,
            "speech_length": len(speech_tokens),
        }


class TrainingCollator:
    """
    Collate training batches with padding and attention masks.
    
    Handles:
    - Padding to max sequence length
    - Attention mask creation (1 for real, 0 for padding)
    - Label mask preservation (-100 for padding, not just for text)
    """
    
    def __init__(self, max_length: int, pad_token_id: int = 0):
        """
        Initialize collator.
        
        Args:
            max_length: Maximum sequence length for padding
            pad_token_id: Token ID for padding (typically 0)
        """
        self.max_length = max_length
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Collate batch into padded tensors.
        
        Args:
            batch: List of examples from dataset
        
        Returns:
            Dict with padded tensors:
            - input_ids: Shape (batch_size, max_length)
            - attention_mask: Shape (batch_size, max_length)
            - labels: Shape (batch_size, max_length), with -100 for ignored positions
        """
        batch_size = len(batch)
        
        # Determine actual max length in batch (not always full max_length)
        actual_max_len = min(
            max(len(ex["input_ids"]) for ex in batch),
            self.max_length,
        )
        
        # Initialize output arrays
        input_ids = np.full(
            (batch_size, actual_max_len),
            self.pad_token_id,
            dtype=np.int32,
        )
        labels = np.full(
            (batch_size, actual_max_len),
            -100,  # Ignore padding positions in loss
            dtype=np.int32,
        )
        attention_mask = np.zeros(
            (batch_size, actual_max_len),
            dtype=np.int32,
        )
        
        # Fill in batch examples
        for i, example in enumerate(batch):
            ids = example["input_ids"][:actual_max_len]
            lbl = example["labels"][:actual_max_len]
            
            seq_len = len(ids)
            input_ids[i, :seq_len] = ids
            labels[i, :seq_len] = lbl
            attention_mask[i, :seq_len] = 1
        
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "labels": torch.from_numpy(labels),
        }


def create_training_dataset(
    config: Optional[TrainingConfig] = None,
    tokenizer: Optional[Any] = None,
) -> tuple:
    """
    Create training dataset and collator.
    
    Args:
        config: TrainingConfig (uses defaults if None)
        tokenizer: HF tokenizer (loads if None)
    
    Returns:
        Tuple of (dataset, collator, config)
    """
    if config is None:
        config = TrainingConfig()
    
    dataset = BodhanTrainingDataset(config=config, tokenizer=tokenizer)
    collator = TrainingCollator(max_length=config.max_seq_length)
    
    return dataset, collator, config
