"""
Preprocessing module for Marathi TTS data pipeline.

Handles:
- Text validation (Marathi Devanagari)
- Audio loading and resampling (48 kHz → 24 kHz)
- SNAC encoding
- Bodhan token serialization

All operations are CPU-compatible and memory-efficient.
"""

import warnings
import numpy as np
import torch
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger(__name__)


class MarathiTextValidator:
    """Validate Marathi language text."""
    
    # Marathi Devanagari Unicode range (Basic, Extended, Vedic)
    DEVANAGARI_RANGES = [
        (0x0900, 0x097F),  # Basic Devanagari
        (0xA8E0, 0xA8FF),  # Devanagari Extended
        (0x1CD0, 0x1CF9),  # Vedic Extensions
    ]
    
    # Common non-Devanagari but acceptable characters
    ALLOWED_ASCII = {' ', '\t', '\n', '.', ',', '!', '?', ';', ':', '"', "'", '-', '(', ')'}
    
    @staticmethod
    def is_devanagari_char(char: str) -> bool:
        """Check if character is Devanagari or allowed ASCII."""
        code = ord(char)
        
        # Check if in Devanagari ranges
        for start, end in MarathiTextValidator.DEVANAGARI_RANGES:
            if start <= code <= end:
                return True
        
        # Check if allowed ASCII
        if char in MarathiTextValidator.ALLOWED_ASCII:
            return True
        
        return False
    
    @staticmethod
    def validate(text: str, min_length: int = 3) -> tuple[bool, str]:
        """
        Validate Marathi text.
        
        Args:
            text: Input text to validate
            min_length: Minimum number of characters
        
        Returns:
            (is_valid, reason)
        """
        if not text:
            return False, "Text is empty"
        
        text = text.strip()
        
        if len(text) < min_length:
            return False, f"Text too short ({len(text)} < {min_length})"
        
        # Check for Devanagari characters
        devanagari_count = sum(1 for c in text if MarathiTextValidator.is_devanagari_char(c))
        total_count = len(text)
        
        if devanagari_count == 0:
            return False, "No Devanagari characters found"
        
        devanagari_ratio = devanagari_count / total_count
        if devanagari_ratio < 0.5:
            return False, f"Low Devanagari ratio ({devanagari_ratio:.1%})"
        
        return True, "Valid Marathi text"


class AudioResampler:
    """Resample audio from 48 kHz to 24 kHz."""
    
    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int = 48000, target_sr: int = 24000) -> np.ndarray:
        """
        Resample audio from 48 kHz to 24 kHz using scipy.
        
        Args:
            audio: float32 mono waveform
            orig_sr: Original sample rate (48000)
            target_sr: Target sample rate (24000)
        
        Returns:
            Resampled float32 mono waveform
        """
        if orig_sr == target_sr:
            return audio
        
        from scipy import signal
        
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        num_samples = int(len(audio) * ratio)
        
        # Use Fourier method for high quality
        resampled = signal.resample(audio, num_samples)
        
        return resampled.astype(np.float32)


class AudioLoader:
    """Load audio from dataset examples using librosa as fallback."""
    
    @staticmethod
    def load_audio(audio_dict: dict) -> tuple[np.ndarray, int]:
        """
        Load audio from dataset example, handling both arrays and file paths.
        
        Supports:
        - Pre-decoded array format (dict with 'array' and 'sampling_rate')
        - Raw format with bytes (dict with 'bytes')
        - Raw format with file path (dict with 'path')
        - Uses librosa for file/bytes loading
        
        Args:
            audio_dict: Dictionary with audio data or path
        
        Returns:
            Tuple of (audio_array, sampling_rate)
        
        Raises:
            ValueError: If audio cannot be loaded
        """
        if not isinstance(audio_dict, dict):
            raise ValueError(f"Expected dict, got {type(audio_dict)}")
        
        # Case 1: Already decoded array with sample rate (highest priority)
        audio_array = audio_dict.get("array")
        sampling_rate = audio_dict.get("sampling_rate")
        
        if audio_array is not None and sampling_rate is not None:
            # Already loaded
            return np.array(audio_array, dtype=np.float32), int(sampling_rate)
        
        # Case 2: Bytes provided (from raw arrow format - try this before path since path may be relative)
        audio_bytes = audio_dict.get("bytes")
        if audio_bytes is not None:
            try:
                import librosa
                import io
                logger.info("Loading audio from bytes")
                audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
                return audio.astype(np.float32), int(sr)
            except Exception as e:
                logger.warning(f"Failed to load from bytes: {e}")
        
        # Case 3: File path provided (fallback if bytes not available or failed)
        file_path = audio_dict.get("path")
        if file_path:
            try:
                import librosa
                logger.info(f"Loading audio from file: {file_path}")
                audio, sr = librosa.load(file_path, sr=None, mono=True)
                return audio.astype(np.float32), int(sr)
            except Exception as e:
                logger.warning(f"Failed to load from path {file_path}: {e}")
        
        # Debug: print available keys
        available_keys = list(audio_dict.keys())
        raise ValueError(
            f"Cannot load audio: no 'array', 'bytes', or working 'path' field. "
            f"Available keys: {available_keys}"
        )





class SNACEncoder:
    """Encode audio using SNAC 1.2.1."""
    
    _snac_model = None  # Cache model to avoid reloading
    
    @staticmethod
    def encode(audio_24khz: np.ndarray, device: str = "cpu") -> dict:
        """
        Encode 24 kHz audio to SNAC codes.
        
        Args:
            audio_24khz: float32 mono waveform at 24 kHz
            device: torch device ("cpu" or "cuda")
        
        Returns:
            Dictionary with c0, c1, c2 codes and metadata
        """
        from snac import SNAC
        
        # Load model once
        if SNACEncoder._snac_model is None:
            logger.info(f"Loading SNAC from hubertsiuzdak/snac_24khz on {device}...")
            SNACEncoder._snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        
        snac = SNACEncoder._snac_model
        
        # Prepare input: shape (1, 1, time) for batch, channel, time
        wav_tensor = torch.from_numpy(audio_24khz).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Encode
        with torch.no_grad():
            codes = snac.encode(wav_tensor)
        
        if not isinstance(codes, (list, tuple)) or len(codes) != 3:
            raise ValueError(f"Expected 3 hierarchical levels, got {len(codes)}")
        
        # Extract and convert to numpy
        c0_raw, c1_raw, c2_raw = codes
        
        c0 = c0_raw.squeeze(0).cpu().numpy().astype(np.int32)
        c1 = c1_raw.squeeze(0).cpu().numpy().astype(np.int32)
        c2 = c2_raw.squeeze(0).cpu().numpy().astype(np.int32)
        
        n_frames = c0.shape[0]
        
        # Verify frame structure
        if c1.shape[0] != n_frames * 2:
            raise ValueError(f"c1 shape mismatch: {c1.shape[0]} != {n_frames * 2}")
        if c2.shape[0] != n_frames * 4:
            raise ValueError(f"c2 shape mismatch: {c2.shape[0]} != {n_frames * 4}")
        
        return {
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "n_frames": n_frames,
            "device": device,
        }


class BodhanTokenSerializer:
    """Serialize SNAC codes to Bodhan token IDs."""
    
    BASE_AUDIO_TOKEN = 128266  # Bodhan audio token base offset
    
    @staticmethod
    def serialize(codes: dict) -> np.ndarray:
        """
        Serialize SNAC hierarchical codes to Bodhan token IDs.
        
        Per-frame ordering: [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]]
        
        Position-based offsets:
        - Position 0: base + 0×4096 + c0[i]
        - Position 1: base + 1×4096 + c1[2i]
        - Position 2: base + 2×4096 + c2[4i]
        - Position 3: base + 3×4096 + c2[4i+1]
        - Position 4: base + 4×4096 + c1[2i+1]
        - Position 5: base + 5×4096 + c2[4i+2]
        - Position 6: base + 6×4096 + c2[4i+3]
        
        Args:
            codes: Dictionary with c0, c1, c2 numpy arrays
        
        Returns:
            Array of token IDs
        """
        c0 = codes["c0"]
        c1 = codes["c1"]
        c2 = codes["c2"]
        n_frames = codes["n_frames"]
        
        tokens = []
        for i in range(n_frames):
            frame_tokens = [
                BodhanTokenSerializer.BASE_AUDIO_TOKEN + 0 * 4096 + int(c0[i]),
                BodhanTokenSerializer.BASE_AUDIO_TOKEN + 1 * 4096 + int(c1[2*i]),
                BodhanTokenSerializer.BASE_AUDIO_TOKEN + 2 * 4096 + int(c2[4*i]),
                BodhanTokenSerializer.BASE_AUDIO_TOKEN + 3 * 4096 + int(c2[4*i+1]),
                BodhanTokenSerializer.BASE_AUDIO_TOKEN + 4 * 4096 + int(c1[2*i+1]),
                BodhanTokenSerializer.BASE_AUDIO_TOKEN + 5 * 4096 + int(c2[4*i+2]),
                BodhanTokenSerializer.BASE_AUDIO_TOKEN + 6 * 4096 + int(c2[4*i+3]),
            ]
            tokens.extend(frame_tokens)
        
        tokens_array = np.array(tokens, dtype=np.int32)
        
        # Validate token ranges
        for pos in range(7):
            expected_min = BodhanTokenSerializer.BASE_AUDIO_TOKEN + pos * 4096
            expected_max = expected_min + 4095
            pos_tokens = tokens_array[pos::7]
            
            if np.any(pos_tokens < expected_min) or np.any(pos_tokens > expected_max):
                raise ValueError(
                    f"Position {pos} tokens out of range: "
                    f"expected [{expected_min}, {expected_max}], "
                    f"got [{pos_tokens.min()}, {pos_tokens.max()}]"
                )
        
        return tokens_array


class DataPipelineProcessor:
    """End-to-end processing of one dataset example."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize processor.
        
        Args:
            device: torch device ("cpu" or "cuda")
        """
        self.device = device
        self.snac_encoder = SNACEncoder()
        self.text_validator = MarathiTextValidator()
        self.audio_loader = AudioLoader()
        self.resampler = AudioResampler()
        self.serializer = BodhanTokenSerializer()
    
    def process(self, example: dict) -> dict:
        """
        Process one dataset example through the full pipeline.
        
        Args:
            example: Dataset example with 'text' and 'audio' fields
                    audio should have 'array' (waveform) and 'sampling_rate'
        
        Returns:
            Dictionary with:
            - text: Validated Marathi text
            - audio_24k: Resampled 24 kHz waveform
            - audio_duration_sec: Duration in seconds
            - codes: SNAC codes (c0, c1, c2)
            - n_frames: Number of SNAC frames
            - tokens: Serialized Bodhan token IDs
            - n_tokens: Number of tokens
        
        Raises:
            ValueError: If validation fails at any step
        """
        # Step 1: Validate text
        text = example.get("text", "")
        
        # Handle text as list (from raw arrow format)
        if isinstance(text, list):
            text = text[0] if text else ""
        
        text = text.strip()
        is_valid, reason = self.text_validator.validate(text)
        if not is_valid:
            raise ValueError(f"Text validation failed: {reason}")
        
        # Step 2: Load and validate audio
        audio_dict = example.get("audio", {})
        
        # Handle audio as list (from raw arrow format)
        if isinstance(audio_dict, list):
            audio_dict = audio_dict[0] if audio_dict else {}
        
        if not isinstance(audio_dict, dict):
            raise ValueError(f"Audio field must be dict, got {type(audio_dict)}")
        
        # Load audio using librosa as fallback for torchcodec
        try:
            audio, orig_sr = self.audio_loader.load_audio(audio_dict)
        except ValueError as e:
            raise ValueError(f"Audio loading failed: {e}")
        
        if audio is None or len(audio) == 0:
            raise ValueError("Audio array is empty")
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed (ensure [-1, 1])
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / (max_val + 1e-8)
        
        # Step 3: Resample to 24 kHz
        audio_24k = self.resampler.resample(audio, orig_sr=int(orig_sr), target_sr=24000)
        audio_duration = len(audio_24k) / 24000.0
        
        # Step 4: Encode with SNAC
        codes = self.snac_encoder.encode(audio_24k, device=self.device)
        n_frames = codes["n_frames"]
        
        # Step 5: Serialize to Bodhan tokens
        tokens = self.serializer.serialize(codes)
        n_tokens = len(tokens)
        
        return {
            "text": text,
            "audio_24k": audio_24k,
            "audio_duration_sec": audio_duration,
            "codes": codes,
            "n_frames": n_frames,
            "tokens": tokens,
            "n_tokens": n_tokens,
        }
