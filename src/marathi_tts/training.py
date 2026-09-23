"""
Training dataset and collation utilities for Bodhan fine-tuning on Marathi TTS.

This module handles:
- Loading Marathi audio + text pairs
- Building Bodhan prompts with speaker/style
- Tokenizing text and speech tokens
- Creating causal LM labels (supervise only speech tokens)
- Enforcing sequence-length constraints
- Batch collation with padding and attention masks

Important:
The public Bodhan repository provides inference-side materials, but does not
provide an official fine-tuning recipe. The training interface implemented here
is therefore a reconstructed training pipeline built around the released Bodhan
tokenization/SNAC contract.
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
import torch

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


# ============================================================================
# Bodhan prompt construction
# ============================================================================


class BodhanPromptBuilder:
    """
    Build Bodhan prompt strings following the inference.py template.

    The prompt intentionally stops at <|start_of_ai|>.

    <|start_of_speech|> is NOT part of the prompt context. It is the first
    generated speech token and is therefore added explicitly to the training
    sequence and supervised by the causal LM loss.
    """

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

        Args:
            text:
                Marathi text to synthesize.

            speaker:
                Speaker name, e.g. "Anagha" or "Chinmay".

            style:
                Emotion/style, e.g. "NEUTRAL" or "HAPPY".

        Returns:
            Prompt string without speech tokens.
        """

        if text is None:
            raise ValueError("text must not be None.")

        text = str(text).strip()

        if not text:
            raise ValueError("text must not be empty.")

        prompt_parts = [
            "<|start_of_human|>",
            "<|begin_of_text|>",
        ]

        if speaker:
            prompt_parts.append(
                f"<|speaker>{speaker}<speaker|>"
            )

        if style:
            prompt_parts.append(
                f"<|style>{style}<style|>"
            )

        # The speech-start token is intentionally excluded here.
        prompt_parts.extend(
            [
                text,
                "<|eot_id|>",
                "<|end_of_human|>",
                "<|start_of_ai|>",
            ]
        )

        return "".join(prompt_parts)


# ============================================================================
# Training dataset
# ============================================================================


class BodhanTrainingDataset:
    """
    Dataset for training Bodhan on Marathi TTS.

    Input:
        Text + Marathi audio at 48 kHz.

    Processing:
        Text validation
        -> Audio resampling (48 -> 24 kHz)
        -> SNAC encoding
        -> Bodhan speech-token serialization
        -> Bodhan prompt construction
        -> Causal-LM training sequence

    Output:
        {
            "input_ids": np.ndarray[int32],
            "labels": np.ndarray[int32],
            "text_length": int,
            "speech_length": int,
            "total_length": int,
        }

    The NumPy representation intentionally remains int32 here because it is
    compact and valid for intermediate preprocessing. The TrainingCollator
    converts token IDs and labels to torch.long before the model forward pass.

    Label semantics:
        - Prompt tokens: -100
        - <|start_of_speech|>: supervised
        - SNAC speech tokens: supervised
        - <|end_of_speech|>: supervised

    Sequence-length invariant:

        text_length
        + 1  (<|start_of_speech|>)
        + speech_length
        + 1  (<|end_of_speech|>)
        <= max_seq_length

    max_audio_tokens provides an independent upper bound on the number of
    serialized Bodhan speech tokens.
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
            config:
                TrainingConfig with dataset and sequence settings.

            tokenizer:
                Hugging Face tokenizer for Bodhan vocab.
                Loaded automatically if None.

            preprocessing_module:
                DataPipelineProcessor for audio/text validation.
                Created automatically if None.
        """

        self.config = config

        self.tokenizer = tokenizer or self._load_tokenizer()

        self.preprocessor = (
            preprocessing_module
            or DataPipelineProcessor(device="cpu")
        )

        self.prompt_builder = BodhanPromptBuilder()

        self.dataset = None

        self._load_dataset()

    # ------------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------------

    def _load_tokenizer(self):
        """Load Bodhan tokenizer from Hugging Face."""

        logger.info(
            "Loading Bodhan tokenizer from bodhan-ai/indic-speak..."
        )

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "bodhan-ai/indic-speak",
                trust_remote_code=True,
            )

            logger.info(
                "Tokenizer loaded. Vocab size: %s",
                tokenizer.vocab_size,
            )

            return tokenizer

        except Exception as e:
            logger.warning(
                "Failed to load Bodhan tokenizer: %s",
                e,
            )

            logger.info(
                "Using placeholder tokenizer for validation mode..."
            )

            return None

    # ------------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------------

    def _load_dataset(self):
        """Load the Marathi TTS dataset."""

        logger.info(
            "Loading %s (%s)...",
            self.config.dataset_name,
            self.config.dataset_split,
        )

        try:
            from datasets import load_dataset

            # Keep the current non-streaming behavior for compatibility with
            # the existing preprocessing pipeline.
            ds = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split,
                streaming=False,
            )

            # Use raw Arrow examples so that audio decoding is handled by the
            # project's preprocessing pipeline rather than torchcodec.
            ds = ds.with_format("arrow")

            # Limit to a deterministic prefix when requested.
            if self.config.max_samples is not None:

                if self.config.max_samples <= 0:
                    raise ValueError(
                        "max_samples must be > 0 when provided."
                    )

                available_count = len(ds)

                selected_count = min(
                    self.config.max_samples,
                    available_count,
                )

                ds = ds.select(range(selected_count))

                logger.info(
                    "Selected %d examples from %d available examples.",
                    selected_count,
                    available_count,
                )

            self.dataset = ds

            logger.info(
                "Loaded %d examples.",
                len(self.dataset),
            )

        except Exception as e:
            logger.error(
                "Failed to load dataset: %s",
                e,
            )
            raise

    # ------------------------------------------------------------------------
    # Text tokenization
    # ------------------------------------------------------------------------

    def _tokenize_prompt(self, prompt_str: str) -> list:
        """
        Tokenize the Bodhan prompt.

        A lightweight placeholder mode is retained for validation tests where
        a real tokenizer is intentionally unavailable.
        """

        if self.tokenizer is None:
            return list(
                range(
                    10,
                    10 + len(prompt_str.split()),
                )
            )

        encoding = self.tokenizer(
            prompt_str,
            add_special_tokens=False,
            return_tensors=None,
            truncation=True,
            max_length=self.config.max_text_length,
        )

        token_ids = encoding["input_ids"]

        if not token_ids:
            raise ValueError(
                "Prompt tokenization produced an empty sequence."
            )

        return token_ids

    # ------------------------------------------------------------------------
    # Speech token preparation
    # ------------------------------------------------------------------------

    def _prepare_speech_tokens(
        self,
        speech_tokens: Any,
        text_length: int,
        idx: int,
    ) -> np.ndarray:
        """
        Apply configured speech-token limits.

        Two independent constraints are enforced:

        1. max_audio_tokens:
            Maximum number of serialized Bodhan speech tokens.

        2. max_seq_length:
            Total sequence must fit:

                text + start_speech + speech + end_speech

        Returns:
            np.ndarray containing the final speech-token sequence.
        """

        # Normalize to a NumPy array so slicing and length handling are
        # deterministic regardless of whether preprocessing returns a list,
        # NumPy array, or tensor.
        if isinstance(speech_tokens, torch.Tensor):
            speech_tokens = (
                speech_tokens.detach()
                .cpu()
                .numpy()
            )

        speech_tokens = np.asarray(
            speech_tokens,
            dtype=np.int32,
        )

        if speech_tokens.ndim != 1:
            speech_tokens = speech_tokens.reshape(-1)

        original_speech_length = len(speech_tokens)

        if original_speech_length == 0:
            raise ValueError(
                f"Example {idx}: preprocessing produced zero speech tokens."
            )

        # Reserve exactly two positions:
        #   1. <|start_of_speech|>
        #   2. <|end_of_speech|>
        sequence_available = (
            self.config.max_seq_length
            - text_length
            - 2
        )

        if sequence_available <= 0:
            raise ValueError(
                f"Example {idx}: prompt is too long. "
                f"text_length={text_length}, "
                f"max_seq_length={self.config.max_seq_length}. "
                "There is no room for speech start/end tokens."
            )

        # max_audio_tokens is an independent upper bound.
        max_speech_tokens = min(
            self.config.max_audio_tokens,
            sequence_available,
        )

        if max_speech_tokens <= 0:
            raise ValueError(
                f"Example {idx}: max_audio_tokens must allow at least "
                "one speech token."
            )

        if original_speech_length > max_speech_tokens:
            speech_tokens = speech_tokens[:max_speech_tokens]

            logger.warning(
                "Example %d: truncating speech tokens from %d to %d "
                "(max_audio_tokens=%d, sequence_available=%d).",
                idx,
                original_speech_length,
                len(speech_tokens),
                self.config.max_audio_tokens,
                sequence_available,
            )

        return speech_tokens

    # ------------------------------------------------------------------------
    # Example construction
    # ------------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get one training example.

        Returns:
            Dict containing:

            input_ids:
                Bodhan prompt + speech tokens.

            labels:
                -100 for prompt positions and real token IDs for the
                supervised speech region.

            text_length:
                Number of prompt tokens.

            speech_length:
                Number of serialized Bodhan speech tokens.

            total_length:
                Total unpadded sequence length.
        """

        example = self.dataset[idx]

        # Convert PyArrow structures to a normal Python dictionary when
        # necessary.
        if hasattr(example, "to_pydict"):
            example = example.to_pydict()

        # Run the existing audio/text preprocessing pipeline.
        try:
            result = self.preprocessor.process(example)

        except Exception as e:
            raise ValueError(
                f"Preprocessing failed for example {idx}: {e}"
            ) from e

        if "text" not in result:
            raise KeyError(
                f"Preprocessing result for example {idx} "
                "does not contain 'text'."
            )

        if "tokens" not in result:
            raise KeyError(
                f"Preprocessing result for example {idx} "
                "does not contain 'tokens'."
            )

        text = result["text"]
        speech_tokens = result["tokens"]

        # Build the textual Bodhan prompt.
        prompt_str = self.prompt_builder.build_prompt(
            text=text,
            speaker=self.config.default_speaker,
            style=self.config.default_style,
        )

        # Tokenize the prompt.
        prompt_token_ids = self._tokenize_prompt(
            prompt_str
        )

        text_length = len(prompt_token_ids)

        # Enforce speech-token limits.
        speech_tokens = self._prepare_speech_tokens(
            speech_tokens=speech_tokens,
            text_length=text_length,
            idx=idx,
        )

        speech_length = len(speech_tokens)

        # Construct:
        #
        #   prompt
        #   + <|start_of_speech|>
        #   + SNAC speech tokens
        #   + <|end_of_speech|>
        #
        # The speech-start token is supervised because the model must learn
        # to transition from <|start_of_ai|> into speech generation.
        input_ids = (
            list(prompt_token_ids)
            + [BODHAN_START_OF_SPEECH]
            + speech_tokens.tolist()
            + [BODHAN_END_OF_SPEECH]
        )

        # Prompt positions are context only and therefore ignored by the loss.
        #
        # Speech start, speech tokens, and speech end are all supervised.
        labels = (
            [-100] * text_length
            + [BODHAN_START_OF_SPEECH]
            + speech_tokens.tolist()
            + [BODHAN_END_OF_SPEECH]
        )

        # The two sequences must always have identical lengths.
        if len(input_ids) != len(labels):
            raise RuntimeError(
                f"Example {idx}: input/label length mismatch: "
                f"{len(input_ids)} != {len(labels)}"
            )

        total_length = len(input_ids)

        # Hard invariant: dataset examples must never exceed the configured
        # model sequence length.
        if total_length > self.config.max_seq_length:
            raise RuntimeError(
                f"Example {idx}: constructed sequence length "
                f"{total_length} exceeds max_seq_length="
                f"{self.config.max_seq_length}."
            )

        # Additional consistency check.
        expected_length = (
            text_length
            + 1
            + speech_length
            + 1
        )

        if total_length != expected_length:
            raise RuntimeError(
                f"Example {idx}: sequence-length accounting mismatch: "
                f"total={total_length}, expected={expected_length}"
            )

        # Validate token ranges when using the real tokenizer.
        if self.tokenizer is not None:

            vocab_size = len(self.tokenizer)

            for token_id in input_ids:
                if token_id < 0 or token_id >= vocab_size:
                    raise ValueError(
                        f"Example {idx}: input token ID {token_id} "
                        f"is outside tokenizer vocabulary [0, {vocab_size})."
                    )

            for label_id in labels:
                if label_id == -100:
                    continue

                if label_id < 0 or label_id >= vocab_size:
                    raise ValueError(
                        f"Example {idx}: label token ID {label_id} "
                        f"is outside tokenizer vocabulary [0, {vocab_size})."
                    )

        return {
            "input_ids": np.asarray(
                input_ids,
                dtype=np.int32,
            ),
            "labels": np.asarray(
                labels,
                dtype=np.int32,
            ),
            "text_length": text_length,
            "speech_length": speech_length,
            "total_length": total_length,
        }

    def __len__(self) -> int:
        """Return dataset size."""

        return (
            len(self.dataset)
            if self.dataset is not None
            else 0
        )


# ============================================================================
# Batch collation
# ============================================================================


class TrainingCollator:
    """
    Collate training batches with padding and attention masks.

    Responsibilities:
    - Pad examples to the longest sequence in the batch.
    - Never silently truncate examples.
    - Create attention masks.
    - Preserve -100 label masking for prompt and padding positions.
    - Return model-compatible torch.long token tensors.

    The dataset is responsible for enforcing max_seq_length. The collator
    therefore treats an over-length example as an error rather than silently
    deleting training tokens.
    """

    def __init__(
        self,
        max_length: int,
        pad_token_id: int = 128263,
    ):
        """
        Initialize collator.

        Args:
            max_length:
                Maximum permitted sequence length.

            pad_token_id:
                Token ID used for input padding.

                Bodhan's verified tokenizer uses 128263 as its pad token.
        """

        if max_length <= 0:
            raise ValueError(
                "max_length must be > 0."
            )

        if pad_token_id < 0:
            raise ValueError(
                "pad_token_id must be >= 0."
            )

        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        batch: list,
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch into padded tensors.

        Args:
            batch:
                List of dataset examples.

        Returns:
            Dict with:

                input_ids:
                    torch.long token IDs.

                attention_mask:
                    torch.long attention mask.

                labels:
                    torch.long label tensor with -100 ignored positions.

        Raises:
            ValueError:
                If any example exceeds max_length.
        """

        if not batch:
            raise ValueError(
                "TrainingCollator received an empty batch."
            )

        # Validate every example before allocating the batch.
        lengths = []

        for i, example in enumerate(batch):

            if "input_ids" not in example:
                raise ValueError(
                    f"Batch example {i}: missing input_ids."
                )

            if "labels" not in example:
                raise ValueError(
                    f"Batch example {i}: missing labels."
                )

            input_length = len(
                example["input_ids"]
            )

            label_length = len(
                example["labels"]
            )

            if input_length != label_length:
                raise ValueError(
                    f"Batch example {i}: input_ids length "
                    f"{input_length} != labels length {label_length}."
                )

            if input_length == 0:
                raise ValueError(
                    f"Batch example {i}: empty sequence."
                )

            if input_length > self.max_length:
                raise ValueError(
                    f"Batch example {i}: sequence length {input_length} "
                    f"exceeds collator max_length={self.max_length}. "
                    "The dataset should enforce max_seq_length before "
                    "collation."
                )

            lengths.append(input_length)

        actual_max_len = max(lengths)

        batch_size = len(batch)

        # --------------------------------------------------------------------
        # Allocate padded NumPy arrays.
        #
        # These are int32 only as intermediate storage. They are explicitly
        # converted to torch.long below because the Hugging Face causal-LM
        # loss requires class-index labels to be int64.
        # --------------------------------------------------------------------

        input_ids = np.full(
            (batch_size, actual_max_len),
            self.pad_token_id,
            dtype=np.int32,
        )

        labels = np.full(
            (batch_size, actual_max_len),
            -100,
            dtype=np.int32,
        )

        attention_mask = np.zeros(
            (batch_size, actual_max_len),
            dtype=np.int32,
        )

        # Copy each example into its unpadded prefix.
        for i, example in enumerate(batch):

            ids = np.asarray(
                example["input_ids"],
                dtype=np.int32,
            )

            lbl = np.asarray(
                example["labels"],
                dtype=np.int32,
            )

            seq_len = len(ids)

            input_ids[
                i,
                :seq_len,
            ] = ids

            labels[
                i,
                :seq_len,
            ] = lbl

            attention_mask[
                i,
                :seq_len,
            ] = 1

        # --------------------------------------------------------------------
        # IMPORTANT:
        #
        # Hugging Face causal LM loss eventually calls
        # torch.nn.functional.cross_entropy().
        #
        # Its class-index targets must be torch.long/int64.
        #
        # Therefore labels MUST NOT remain torch.int32.
        # --------------------------------------------------------------------

        return {
            "input_ids": torch.from_numpy(
                input_ids
            ).to(dtype=torch.long),

            "attention_mask": torch.from_numpy(
                attention_mask
            ).to(dtype=torch.long),

            "labels": torch.from_numpy(
                labels
            ).to(dtype=torch.long),
        }


# ============================================================================
# Dataset factory
# ============================================================================


def create_training_dataset(
    config: Optional[TrainingConfig] = None,
    tokenizer: Optional[Any] = None,
) -> tuple:
    """
    Create training dataset and collator.

    Args:
        config:
            TrainingConfig. Uses defaults if None.

        tokenizer:
            HF tokenizer. Loads automatically if None.

    Returns:
        Tuple:

            (dataset, collator, config)
    """

    if config is None:
        config = TrainingConfig()

    dataset = BodhanTrainingDataset(
        config=config,
        tokenizer=tokenizer,
    )

    # Prefer the actual tokenizer's pad token when available.
    #
    # Bodhan's tokenizer was verified to expose:
    #
    #     pad_token_id = 128263
    #
    # Falling back to 128263 keeps the training contract explicit even if
    # create_training_dataset() is called without a tokenizer object.
    if tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id

        if pad_token_id is None:
            raise ValueError(
                "Bodhan tokenizer does not define pad_token_id."
            )
    elif dataset.tokenizer is not None:
        pad_token_id = dataset.tokenizer.pad_token_id

        if pad_token_id is None:
            raise ValueError(
                "Bodhan tokenizer does not define pad_token_id."
            )
    else:
        pad_token_id = 128263

    collator = TrainingCollator(
        max_length=config.max_seq_length,
        pad_token_id=int(pad_token_id),
    )

    return dataset, collator, config