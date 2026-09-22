# Marathi TTS Data Pipeline

## Overview

The data pipeline transforms raw IndicTTS_Marathi dataset examples into Bodhan-compatible training data.

**Input:** Dataset example (48 kHz Marathi audio + Devanagari text)  
**Output:** (Marathi text, Bodhan speech tokens) training pairs

---

## Pipeline Stages

### 1. Text Validation

**Module:** `MarathiTextValidator`

Validates Marathi language text:
- Minimum length: 3 characters
- Devanagari script presence: ≥50% of text
- Allowed characters: Devanagari Unicode ranges (0x0900-0x097F, 0xA8E0-0xA8FF, 0x1CD0-0x1CF9)
- Accepts common ASCII (space, punctuation)

**Handles:**
- Empty/whitespace-only text → reject
- Non-Marathi text → reject
- Mixed script text → accept if ≥50% Devanagari

### 2. Audio Resampling

**Module:** `AudioResampler`

Converts 48 kHz input audio to 24 kHz (Bodhan/SNAC requirement):
- Input: 48 kHz mono WAV from IndicTTS_Marathi
- Method: `scipy.signal.resample()` (Fourier-based, high-quality)
- Output: 24 kHz mono float32
- Duration scaling: 48 kHz → 24 kHz = 0.5× audio length

**Example:**
```
48 kHz: 10.33 hours → 24 kHz: ~5.17 hours after resampling
```

### 3. Audio Normalization

**In `DataPipelineProcessor.process()`**

- Ensures float32 dtype
- Normalizes to [-1, 1] range (no clipping)

### 4. SNAC Encoding

**Module:** `SNACEncoder`

Encodes 24 kHz waveform using SNAC 1.2.1 (Speech Neural Audio Codec):
- Input: 24 kHz mono float32 waveform
- Output: Hierarchical codes (c0, c1, c2)
- Model: `hubertsiuzdak/snac_24khz` from Hugging Face
- Processing: Loaded once, cached for memory efficiency

**Hierarchical structure:**
- **c0 (Level 0):** 1 code per frame @ 12.5 Hz
- **c1 (Level 1):** 2 codes per frame @ 25 Hz
- **c2 (Level 2):** 4 codes per frame @ 50 Hz
- **Total:** 7 codes per frame (1 + 2 + 4)

**Frame rate:** 1 frame per 80 audio samples @ 24 kHz = 300 frames/second

### 5. Bodhan Token Serialization

**Module:** `BodhanTokenSerializer`

Converts SNAC hierarchical codes to Bodhan token IDs.

**Serialization formula (per frame i):**
```
Position 0: base + 0×4096 + c0[i]
Position 1: base + 1×4096 + c1[2i]
Position 2: base + 2×4096 + c2[4i]
Position 3: base + 3×4096 + c2[4i+1]
Position 4: base + 4×4096 + c1[2i+1]
Position 5: base + 5×4096 + c2[4i+2]
Position 6: base + 6×4096 + c2[4i+3]
```

**Token range:** Base = 128,266 + position offset (0-6) × 4,096 + code value (0-4,095)
- Position 0: [128,266 - 132,361]
- Position 1: [132,362 - 136,457]
- Position 2: [136,458 - 140,553]
- Position 3: [140,554 - 144,649]
- Position 4: [144,650 - 148,745]
- Position 5: [148,746 - 152,841]
- Position 6: [152,842 - 156,937]

**Output:** Array of 7N token IDs (N = number of frames)

---

## End-to-End Processor

**Class:** `DataPipelineProcessor`

Single-call processing of one dataset example:

```python
from marathi_tts.preprocess import DataPipelineProcessor

processor = DataPipelineProcessor(device="cpu")

result = processor.process(example)
# Returns:
# {
#     "text": "वाक्य...",
#     "audio_24k": np.ndarray,          # 24 kHz waveform
#     "audio_duration_sec": float,      # Duration in seconds
#     "codes": {"c0": ..., "c1": ..., "c2": ...},
#     "n_frames": int,                  # SNAC frame count
#     "tokens": np.ndarray,             # Bodhan token IDs
#     "n_tokens": int,                  # Total token count
# }
```

---

## Validation & Testing

**Script:** `scripts/validate_data_pipeline.py`

Validates the complete pipeline on 10 IndicTTS_Marathi examples:

```bash
python scripts/validate_data_pipeline.py
```

**Checks:**
- All text fields are valid Marathi
- All audio loads, resamples, and encodes correctly
- SNAC produces valid hierarchical codes
- Token serialization produces valid Bodhan token IDs
- No data loss or corruption

**Output:**
- Per-example: text preview, duration, frame count, token count
- Summary: total processed, failures, statistics
- Pass/fail verdict

---

## CPU-Memory Considerations

- **SNAC Model:** Cached after first load (~500 MB in memory)
- **Per Example:** Audio + codes + tokens ≈ 10-50 MB peak
- **Batch Processing:** Process examples individually; no large batches in memory
- **Resampling:** Uses FFT-based method (scipy); O(N log N) time, O(N) space

**Typical runtime per example:** 1-2 seconds (CPU, including SNAC)

---

## Known Limitations

1. **Tokenizer Not Applied:** Text is validated but not tokenized (text tokenization deferred to model layer)
2. **No Normalization:** Audio normalized to [-1, 1] but not further (e.g., no loudness targeting)
3. **Single Split:** No train/val/test splitting (applied at dataset loading, not pipeline)
4. **Error Handling:** Raises exceptions on validation failure (no recovery/fallback)

---

## Example Walkthrough

### Input
```json
{
  "text": "नमस्कार",
  "audio": {
    "array": [float32 array, 48000 Hz],
    "sampling_rate": 48000,
    "path": "..."
  }
}
```

### Step 1: Text Validation
```
Text: "नमस्कार"
→ Length: 7 characters
→ Devanagari ratio: 100%
→ Result: VALID
```

### Step 2: Audio Resampling
```
Input:  48 kHz, 96000 samples → 2.0 seconds
Resample: 48000 / 24000 = 2:1
Output: 24 kHz, 48000 samples → 2.0 seconds
```

### Step 3: SNAC Encoding
```
Audio: 48000 samples @ 24 kHz
SNAC: 1 frame per 80 samples = 600 frames
Output: c0 shape (600,), c1 shape (1200,), c2 shape (2400,)
```

### Step 4: Bodhan Serialization
```
600 frames × 7 tokens/frame = 4200 tokens
Tokens: [128266, 132400, 140500, ..., 156900]
```

### Output
```python
{
  "text": "नमस्कार",
  "audio_duration_sec": 2.0,
  "n_frames": 600,
  "tokens": [128266, 132400, ...],  # 4200 token IDs
  "n_tokens": 4200,
}
```

---

## Next Steps (Deferred)

1. **Dataloader:** Load multiple examples, batch tokens with padding
2. **Fine-tuning:** Use (text, speech_tokens) pairs for model training
3. **Evaluation:** Inference and perceptual quality assessment

---

## References

- **SNAC:** hubertsiuzdak/snac_24khz (Hugging Face)
- **Bodhan Tokenization:** Position-based offset mapping (validated in scripts/validate_snac_serialization.py)
- **Dataset:** SPRINGLab/IndicTTS_Marathi
- **Preprocessing Module:** src/marathi_tts/preprocess.py
- **Validation Script:** scripts/validate_data_pipeline.py
