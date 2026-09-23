"""
Preprocessing module for Marathi TTS data pipeline.

Handles:
- Text validation (Marathi Devanagari)
- Audio loading and resampling (48 kHz → 24 kHz)
- SNAC encoding
- Bodhan token serialization

The implementation is designed to work with both:
- Hugging Face decoded audio dictionaries
- Hugging Face torchcodec AudioDecoder objects

All operations are CPU-compatible except SNAC encoding, which can
optionally use CUDA when available.
"""

from __future__ import annotations

import io
import logging
import warnings
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Warning / logging configuration
# ---------------------------------------------------------------------------

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Marathi text validation
# ---------------------------------------------------------------------------


class MarathiTextValidator:
    """Validate Marathi language text."""

    # Marathi Devanagari Unicode ranges
    DEVANAGARI_RANGES = [
        (0x0900, 0x097F),  # Basic Devanagari
        (0xA8E0, 0xA8FF),  # Devanagari Extended
        (0x1CD0, 0x1CF9),  # Vedic Extensions
    ]

    # Common non-Devanagari characters accepted in text
    ALLOWED_ASCII = {
        " ",
        "\t",
        "\n",
        ".",
        ",",
        "!",
        "?",
        ";",
        ":",
        '"',
        "'",
        "-",
        "(",
        ")",
    }

    @staticmethod
    def is_devanagari_char(char: str) -> bool:
        """Check if a character is Devanagari or explicitly allowed ASCII."""

        code = ord(char)

        for start, end in MarathiTextValidator.DEVANAGARI_RANGES:
            if start <= code <= end:
                return True

        if char in MarathiTextValidator.ALLOWED_ASCII:
            return True

        return False

    @staticmethod
    def validate(
        text: str,
        min_length: int = 3,
    ) -> tuple[bool, str]:
        """
        Validate Marathi text.

        Args:
            text: Input text to validate.
            min_length: Minimum number of characters.

        Returns:
            Tuple of (is_valid, reason).
        """

        if not text:
            return False, "Text is empty"

        text = text.strip()

        if len(text) < min_length:
            return False, f"Text too short ({len(text)} < {min_length})"

        devanagari_count = sum(
            1
            for char in text
            if MarathiTextValidator.is_devanagari_char(char)
        )

        total_count = len(text)

        if devanagari_count == 0:
            return False, "No Devanagari characters found"

        devanagari_ratio = devanagari_count / total_count

        if devanagari_ratio < 0.5:
            return (
                False,
                f"Low Devanagari ratio ({devanagari_ratio:.1%})",
            )

        return True, "Valid Marathi text"


# ---------------------------------------------------------------------------
# Audio resampling
# ---------------------------------------------------------------------------


class AudioResampler:
    """Resample audio from 48 kHz to 24 kHz."""

    @staticmethod
    def resample(
        audio: np.ndarray,
        orig_sr: int = 48000,
        target_sr: int = 24000,
    ) -> np.ndarray:
        """
        Resample audio using scipy's Fourier resampling method.

        Args:
            audio: Float32 mono waveform.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled float32 mono waveform.
        """

        audio = np.asarray(audio, dtype=np.float32)

        if orig_sr <= 0 or target_sr <= 0:
            raise ValueError(
                f"Sample rates must be positive: "
                f"orig_sr={orig_sr}, target_sr={target_sr}"
            )

        if audio.size == 0:
            raise ValueError("Cannot resample an empty audio array")

        if orig_sr == target_sr:
            return audio.astype(np.float32, copy=False)

        from scipy import signal

        ratio = target_sr / orig_sr
        num_samples = int(round(len(audio) * ratio))

        if num_samples <= 0:
            raise ValueError(
                f"Resampling produced invalid length: {num_samples}"
            )

        # Fourier method.
        resampled = signal.resample(audio, num_samples)

        return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


class AudioLoader:
    """
    Load audio from Hugging Face datasets.

    Supports:
    - Hugging Face torchcodec AudioDecoder
    - Decoded dictionary with array + sampling_rate
    - Dictionary containing bytes
    - Dictionary containing file path
    """

    @staticmethod
    def _convert_to_mono(
        audio: np.ndarray,
    ) -> np.ndarray:
        """Convert an audio array to a float32 mono waveform."""

        audio = np.asarray(audio)

        if audio.size == 0:
            raise ValueError("Decoded audio is empty")

        # Typical torchcodec format is [channels, time].
        if audio.ndim == 2:
            if audio.shape[0] <= 8:
                audio = audio.mean(axis=0)
            else:
                # Defensive handling for [time, channels].
                audio = audio.mean(axis=1)

        elif audio.ndim > 2:
            raise ValueError(
                f"Unsupported audio dimensionality: {audio.shape}"
            )

        return audio.astype(np.float32, copy=False)

    @staticmethod
    def _load_audio_decoder(
        audio_decoder: Any,
    ) -> tuple[np.ndarray, int]:
        """
        Decode a Hugging Face torchcodec AudioDecoder.

        AudioDecoder exposes get_all_samples(), which returns an
        AudioSamples object containing:
            - data
            - sample_rate
        """

        if not hasattr(audio_decoder, "get_all_samples"):
            raise ValueError(
                "Object does not expose get_all_samples(): "
                f"{type(audio_decoder)}"
            )

        logger.debug(
            "Decoding Hugging Face AudioDecoder: %s",
            type(audio_decoder),
        )

        samples = audio_decoder.get_all_samples()

        if not hasattr(samples, "data"):
            raise ValueError(
                "AudioDecoder.get_all_samples() did not return "
                "an object containing 'data'"
            )

        if not hasattr(samples, "sample_rate"):
            raise ValueError(
                "AudioDecoder.get_all_samples() did not return "
                "an object containing 'sample_rate'"
            )

        audio_data = samples.data

        if torch.is_tensor(audio_data):
            audio_data = audio_data.detach().cpu().numpy()
        else:
            audio_data = np.asarray(audio_data)

        audio_data = AudioLoader._convert_to_mono(audio_data)

        sample_rate = int(samples.sample_rate)

        if sample_rate <= 0:
            raise ValueError(
                f"Invalid decoded sample rate: {sample_rate}"
            )

        return audio_data, sample_rate

    @staticmethod
    def load_audio(
        audio_input: Any,
    ) -> tuple[np.ndarray, int]:
        """
        Load audio from a dataset example.

        Supported inputs:

        1. Hugging Face AudioDecoder:
           datasets.features._torchcodec.AudioDecoder

        2. Decoded dictionary:
           {
               "array": waveform,
               "sampling_rate": 48000
           }

        3. Raw dictionary containing:
           {
               "bytes": ...
           }

        4. Dictionary containing:
           {
               "path": ...
           }

        Returns:
            Tuple of (mono float32 waveform, sampling rate).
        """

        # ---------------------------------------------------------------
        # Case 1: Hugging Face torchcodec AudioDecoder
        # ---------------------------------------------------------------

        if hasattr(audio_input, "get_all_samples"):
            return AudioLoader._load_audio_decoder(audio_input)

        # ---------------------------------------------------------------
        # Case 2+: dictionary-based formats
        # ---------------------------------------------------------------

        if not isinstance(audio_input, dict):
            raise ValueError(
                f"Expected audio dictionary or AudioDecoder, "
                f"got {type(audio_input)}"
            )

        # ---------------------------------------------------------------
        # Case 2: Already decoded array
        # ---------------------------------------------------------------

        audio_array = audio_input.get("array")
        sampling_rate = audio_input.get("sampling_rate")

        if audio_array is not None and sampling_rate is not None:
            audio_array = AudioLoader._convert_to_mono(audio_array)

            return (
                audio_array,
                int(sampling_rate),
            )

        # ---------------------------------------------------------------
        # Case 3: Raw bytes
        # ---------------------------------------------------------------

        audio_bytes = audio_input.get("bytes")

        if audio_bytes is not None:
            try:
                import librosa

                logger.info("Loading audio from bytes")

                audio, sr = librosa.load(
                    io.BytesIO(audio_bytes),
                    sr=None,
                    mono=True,
                )

                return (
                    audio.astype(np.float32),
                    int(sr),
                )

            except Exception as exc:
                logger.warning(
                    "Failed to load audio from bytes: %s",
                    exc,
                )

        # ---------------------------------------------------------------
        # Case 4: File path
        # ---------------------------------------------------------------

        file_path = audio_input.get("path")

        if file_path:
            try:
                import librosa

                logger.info(
                    "Loading audio from file: %s",
                    file_path,
                )

                audio, sr = librosa.load(
                    file_path,
                    sr=None,
                    mono=True,
                )

                return (
                    audio.astype(np.float32),
                    int(sr),
                )

            except Exception as exc:
                logger.warning(
                    "Failed to load audio from path %s: %s",
                    file_path,
                    exc,
                )

        available_keys = list(audio_input.keys())

        raise ValueError(
            "Cannot load audio: no 'array', 'bytes', or working "
            "'path' field. "
            f"Available keys: {available_keys}"
        )


# ---------------------------------------------------------------------------
# SNAC encoder
# ---------------------------------------------------------------------------


class SNACEncoder:
    """Encode 24 kHz audio using SNAC."""

    _snac_model = None
    _snac_device = None

    @staticmethod
    def encode(
        audio_24khz: np.ndarray,
        device: str = "cpu",
    ) -> dict:
        """
        Encode 24 kHz audio into hierarchical SNAC codes.

        Expected hierarchy:

            c0: n_frames
            c1: 2 * n_frames
            c2: 4 * n_frames

        Args:
            audio_24khz: Float32 mono waveform at 24 kHz.
            device: "cpu" or "cuda".

        Returns:
            Dictionary containing c0, c1, c2, n_frames and device.
        """

        from snac import SNAC

        audio_24khz = np.asarray(
            audio_24khz,
            dtype=np.float32,
        )

        if audio_24khz.size == 0:
            raise ValueError("Cannot encode empty audio")

        # ---------------------------------------------------------------
        # Load/cache SNAC
        # ---------------------------------------------------------------

        requested_device = str(device)

        if (
            SNACEncoder._snac_model is None
            or SNACEncoder._snac_device != requested_device
        ):
            logger.info(
                "Loading SNAC from hubertsiuzdak/snac_24khz "
                "on %s...",
                requested_device,
            )

            SNACEncoder._snac_model = (
                SNAC.from_pretrained(
                    "hubertsiuzdak/snac_24khz"
                )
                .to(requested_device)
                .eval()
            )

            SNACEncoder._snac_device = requested_device

        snac = SNACEncoder._snac_model

        # ---------------------------------------------------------------
        # Prepare input:
        # [batch, channels, time]
        # ---------------------------------------------------------------

        wav_tensor = (
            torch.from_numpy(audio_24khz)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(requested_device)
        )

        # ---------------------------------------------------------------
        # Encode
        # ---------------------------------------------------------------

        with torch.no_grad():
            codes = snac.encode(wav_tensor)

        if not isinstance(codes, (list, tuple)):
            raise ValueError(
                f"Expected SNAC output list/tuple, got {type(codes)}"
            )

        if len(codes) != 3:
            raise ValueError(
                f"Expected 3 hierarchical SNAC levels, "
                f"got {len(codes)}"
            )

        c0_raw, c1_raw, c2_raw = codes

        c0 = (
            c0_raw.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int32)
        )

        c1 = (
            c1_raw.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int32)
        )

        c2 = (
            c2_raw.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int32)
        )

        # ---------------------------------------------------------------
        # Validate hierarchical structure
        # ---------------------------------------------------------------

        n_frames = int(c0.shape[0])

        if c1.shape[0] != n_frames * 2:
            raise ValueError(
                f"c1 shape mismatch: "
                f"{c1.shape[0]} != {n_frames * 2}"
            )

        if c2.shape[0] != n_frames * 4:
            raise ValueError(
                f"c2 shape mismatch: "
                f"{c2.shape[0]} != {n_frames * 4}"
            )

        return {
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "n_frames": n_frames,
            "device": requested_device,
        }


# ---------------------------------------------------------------------------
# Bodhan SNAC → audio-token serialization
# ---------------------------------------------------------------------------


class BodhanTokenSerializer:
    """
    Serialize SNAC codes to Bodhan audio token IDs.

    IMPORTANT:
    Bodhan uses one global audio-token base. There is no additional
    position-dependent +4096 offset.
    """

    BASE_AUDIO_TOKEN = 128266
    CODEBOOK_SIZE = 4096
    NUM_CODEBOOKS = 7

    @staticmethod
    def serialize(
        codes: dict,
    ) -> np.ndarray:
        """
        Serialize hierarchical SNAC codes using Bodhan's 7-token
        frame ordering.

        For frame i:

            [
                c0[i],
                c1[2*i],
                c2[4*i],
                c2[4*i+1],
                c1[2*i+1],
                c2[4*i+2],
                c2[4*i+3],
            ]

        Each value receives the same global BASE_AUDIO_TOKEN.

        In other words:

            token = 128266 + SNAC_code

        There is NO:

            128266 + position * 4096 + SNAC_code

        position-dependent offset.

        This ordering is the inverse of Bodhan's official
        ids_to_codes() reconstruction.

        Args:
            codes:
                Dictionary containing:
                - c0
                - c1
                - c2
                - n_frames

        Returns:
            np.ndarray of int32 Bodhan audio token IDs.
        """

        c0 = np.asarray(
            codes["c0"]
        ).reshape(-1)

        c1 = np.asarray(
            codes["c1"]
        ).reshape(-1)

        c2 = np.asarray(
            codes["c2"]
        ).reshape(-1)

        n_frames = int(
            codes["n_frames"]
        )

        # ---------------------------------------------------------------
        # Validate hierarchy before indexing
        # ---------------------------------------------------------------

        if len(c0) < n_frames:
            raise ValueError(
                f"c0 has {len(c0)} entries but "
                f"n_frames={n_frames}"
            )

        if len(c1) < 2 * n_frames:
            raise ValueError(
                f"c1 has {len(c1)} entries but "
                f"{2 * n_frames} are required"
            )

        if len(c2) < 4 * n_frames:
            raise ValueError(
                f"c2 has {len(c2)} entries but "
                f"{4 * n_frames} are required"
            )

        # ---------------------------------------------------------------
        # Serialize each SNAC frame
        # ---------------------------------------------------------------

        tokens: list[int] = []

        for i in range(n_frames):
            tokens.extend(
                [
                    # Codebook 0
                    BodhanTokenSerializer.BASE_AUDIO_TOKEN
                    + int(c0[i]),

                    # Codebook 1, first half
                    BodhanTokenSerializer.BASE_AUDIO_TOKEN
                    + int(c1[2 * i]),

                    # Codebook 2, first half
                    BodhanTokenSerializer.BASE_AUDIO_TOKEN
                    + int(c2[4 * i]),

                    BodhanTokenSerializer.BASE_AUDIO_TOKEN
                    + int(c2[4 * i + 1]),

                    # Codebook 1, second half
                    BodhanTokenSerializer.BASE_AUDIO_TOKEN
                    + int(c1[2 * i + 1]),

                    # Codebook 2, second half
                    BodhanTokenSerializer.BASE_AUDIO_TOKEN
                    + int(c2[4 * i + 2]),

                    BodhanTokenSerializer.BASE_AUDIO_TOKEN
                    + int(c2[4 * i + 3]),
                ]
            )

        tokens_array = np.asarray(
            tokens,
            dtype=np.int32,
        )

        # ---------------------------------------------------------------
        # Validate global Bodhan audio-token range
        # ---------------------------------------------------------------

        expected_min = (
            BodhanTokenSerializer.BASE_AUDIO_TOKEN
        )

        expected_max = (
            expected_min
            + (
                BodhanTokenSerializer.NUM_CODEBOOKS
                * BodhanTokenSerializer.CODEBOOK_SIZE
            )
        )

        if len(tokens_array) > 0:

            actual_min = int(
                tokens_array.min()
            )

            actual_max = int(
                tokens_array.max()
            )

            if actual_min < expected_min:
                raise ValueError(
                    "Serialized token below Bodhan audio range: "
                    f"min={actual_min}, "
                    f"expected >= {expected_min}"
                )

            if actual_max >= expected_max:
                raise ValueError(
                    "Serialized token above Bodhan audio range: "
                    f"max={actual_max}, "
                    f"expected < {expected_max}"
                )

        # Every SNAC frame must become exactly seven tokens.
        expected_token_count = n_frames * 7

        if len(tokens_array) != expected_token_count:
            raise ValueError(
                f"Serialized token count mismatch: "
                f"{len(tokens_array)} != "
                f"{expected_token_count}"
            )

        return tokens_array


# ---------------------------------------------------------------------------
# End-to-end data pipeline
# ---------------------------------------------------------------------------


class DataPipelineProcessor:
    """End-to-end processing of one Marathi TTS dataset example."""

    def __init__(
        self,
        device: str = "cpu",
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            device: Device used for SNAC encoding.
                    "cpu" or "cuda".
        """

        self.device = device

        self.snac_encoder = SNACEncoder()
        self.text_validator = MarathiTextValidator()
        self.audio_loader = AudioLoader()
        self.resampler = AudioResampler()
        self.serializer = BodhanTokenSerializer()

    def process(
        self,
        example: dict,
    ) -> dict:
        """
        Process one dataset example through the complete pipeline.

        Pipeline:

            text
              ↓
            Marathi validation
              ↓
            audio decoding
              ↓
            mono waveform
              ↓
            24 kHz resampling
              ↓
            SNAC encoding
              ↓
            Bodhan 7-token serialization

        Args:
            example:
                Dataset example containing "text" and "audio".

        Returns:
            Dictionary containing:
            - text
            - audio_24k
            - audio_duration_sec
            - codes
            - n_frames
            - tokens
            - n_tokens

        Raises:
            ValueError:
                If text, audio, SNAC structure, or tokenization
                validation fails.
        """

        # ---------------------------------------------------------------
        # Step 1: Validate text
        # ---------------------------------------------------------------

        text = example.get(
            "text",
            "",
        )

        # Some raw Arrow configurations may return text as a list.
        if isinstance(text, list):
            text = (
                text[0]
                if text
                else ""
            )

        if not isinstance(text, str):
            text = str(text)

        text = text.strip()

        is_valid, reason = (
            self.text_validator.validate(text)
        )

        if not is_valid:
            raise ValueError(
                f"Text validation failed: {reason}"
            )

        # ---------------------------------------------------------------
        # Step 2: Load audio
        # ---------------------------------------------------------------

        audio_input = example.get(
            "audio",
            {},
        )

        # Some raw Arrow configurations may return a list.
        if isinstance(audio_input, list):
            audio_input = (
                audio_input[0]
                if audio_input
                else {}
            )

        try:
            audio, orig_sr = (
                self.audio_loader.load_audio(
                    audio_input
                )
            )
        except Exception as exc:
            raise ValueError(
                f"Audio loading failed: {exc}"
            ) from exc

        if audio is None or len(audio) == 0:
            raise ValueError(
                "Audio array is empty"
            )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        # ---------------------------------------------------------------
        # Step 2b: Normalize waveform if required
        # ---------------------------------------------------------------

        max_val = float(
            np.abs(audio).max()
        )

        if max_val > 1.0:
            audio = (
                audio
                / (max_val + 1e-8)
            )

        # ---------------------------------------------------------------
        # Step 3: Resample to 24 kHz
        # ---------------------------------------------------------------

        audio_24k = (
            self.resampler.resample(
                audio,
                orig_sr=int(orig_sr),
                target_sr=24000,
            )
        )

        audio_duration = (
            len(audio_24k)
            / 24000.0
        )

        # ---------------------------------------------------------------
        # Step 4: Encode with SNAC
        # ---------------------------------------------------------------

        codes = (
            self.snac_encoder.encode(
                audio_24khz=audio_24k,
                device=self.device,
            )
        )

        n_frames = int(
            codes["n_frames"]
        )

        # ---------------------------------------------------------------
        # Step 5: Serialize to Bodhan tokens
        # ---------------------------------------------------------------

        tokens = (
            self.serializer.serialize(
                codes
            )
        )

        n_tokens = len(tokens)

        # ---------------------------------------------------------------
        # Final structural validation
        # ---------------------------------------------------------------

        expected_tokens = n_frames * 7

        if n_tokens != expected_tokens:
            raise ValueError(
                f"Bodhan token count mismatch: "
                f"{n_tokens} != "
                f"{expected_tokens}"
            )

        if n_tokens % 7 != 0:
            raise ValueError(
                f"Bodhan token sequence is not "
                f"7-token aligned: {n_tokens}"
            )

        # ---------------------------------------------------------------
        # Return processed example
        # ---------------------------------------------------------------

        return {
            "text": text,
            "audio_24k": audio_24k,
            "audio_duration_sec": audio_duration,
            "codes": codes,
            "n_frames": n_frames,
            "tokens": tokens,
            "n_tokens": n_tokens,
        }