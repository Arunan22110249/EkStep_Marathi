# Marathi TTS Dataset Selection

## Assignment Requirements

- **Target Model:** Bodhan Indic-Speak TTS (fine-tuning)
- **Language:** Marathi (मराठी)
- **Audio Format:** Must work with 24 kHz waveforms (upsampled from dataset)
- **Data Pairing:** Audio + text transcripts required
- **Scope:** Research-grade fine-tuning, not production deployment
- **License:** Must permit academic fine-tuning and experimentation
- **Accessibility:** From permitted sources (Hugging Face, AIKosh)
- **Feasibility:** Should download and process on CPU-only machine
- **Size Target:** Enough for meaningful fine-tuning iteration (>500 samples ideal)

---

## Candidate Datasets

### 1. Shunya Labs Marathi Speech Dataset
- **Source:** https://huggingface.co/datasets/shunyalabs/marathi-speech-dataset
- **Language:** Marathi (मराठी)
- **Task:** Speech Recognition / TTS-compatible audio corpus
- **Audio:** Marathi speech samples in parquet format
- **Transcript:** Text transcripts paired with audio
- **Speakers:** Multiple speakers (exact count unknown, not documented)
- **Size:** 352,966 samples | 43.7 GB total | 100K-1M range
- **Splits:** Not documented (single dataset without explicit train/val/test split)
- **License:** Not explicitly stated on dataset card
- **Preprocessing:** Parquet format; may require unpacking and resampling to 24 kHz
- **Suitability:** 
  - ✅ Native Marathi speech corpus (primary language, not multilingual inclusion)
  - ✅ Audio + text pairs verified
  - ✅ Large sample count (352K) supports meaningful fine-tuning
  - ✅ Accessible on Hugging Face with no authentication barrier
  - ⚠️ License and usage terms not clearly documented on dataset card
  - ⚠️ No explicit data splits provided
  - ⚠️ Speaker metadata not documented
  - ⚠️ No sample rate documentation (requires inspection)
- **Concerns:**
  - Missing metadata on licensing and speaker information
  - Unclear whether samples are balanced or have quality guarantees
  - Unknown audio quality and duration statistics

### 2. Google FLEURS (Marathi subset: mr_in)
- **Source:** https://huggingface.co/datasets/google/fleurs
- **Language:** Marathi (मराठी), South-Asia region
- **Task:** Automatic Speech Recognition (ASR) focused, but contains audio/text pairs suitable for TTS
- **Audio:** Speech in WAV format, 16 kHz sampling rate
- **Transcript:** Clean transcriptions (raw and normalized versions provided)
- **Speakers:** Multiple speakers (~3 per utterance), gender-balanced when possible
- **Size:** 
  - Train: 3,270 samples (~10 hours)
  - Validation: 443 samples
  - Test: 1,020 samples
  - Total: ~4,733 samples | ~12 hours of audio
- **Splits:** Explicitly defined (train/validation/test with disjoint speakers)
- **License:** CC-BY 4.0 (permissive, allows academic and research use)
- **Preprocessing:**
  - Requires upsampling from 16 kHz to 24 kHz
  - Text normalization already performed
  - Character-level tokenization available
  - Audio arrays provided directly (no extraction needed)
- **Suitability:**
  - ✅ Part of massively multilingual benchmark (102 languages)
  - ✅ High-quality transcriptions with multiple variants (raw, normalized, character)
  - ✅ Clean CC-BY 4.0 license explicitly documented
  - ✅ Professional data collection and validation pipeline
  - ✅ Pre-existing train/val/test splits with speaker separation
  - ✅ Gender metadata available
  - ✅ Easy loading via Hugging Face datasets library
  - ⚠️ Read speech only (not spontaneous/conversational)
  - ⚠️ Smaller sample count (3.2K training) — may be limited for extensive fine-tuning
  - ⚠️ Shorter utterances (≤30 seconds, domain-specific parallel sentences)
  - ⚠️ 16 kHz baseline requires resampling infrastructure
- **Concerns:**
  - Limited absolute sample size (3K training samples)
  - Wikipedia text domain may not match typical speech synthesis use cases
  - May not provide enough diversity for robust TTS model training

### 3. PAARI Marathi TTS Dataset
- **Source:** https://huggingface.co/datasets/keplersystems/PAARI-Marathi-TTS
- **Language:** Marathi (मराठी)
- **Task:** Text-to-Speech (explicit TTS optimization)
- **Audio:** ❌ **TEXT ONLY** — No audio samples
- **Transcript:** Marathi text chunks (≤800 characters, TTS-optimized chunking)
- **Speakers:** N/A (text only)
- **Size:** 1K-10K range | 21.5 GB
- **Splits:** Single "train" split documented
- **License:** Requires contact information sharing (gated access)
- **Preprocessing:** HTML/entity decoding, Devanagari normalization already applied
- **Suitability:**
  - ❌ **NOT SUITABLE** — Contains text only, no audio samples
  - ✅ Marathi language content
  - ✅ TTS-aware text chunking
  - ✅ Clean preprocessing already applied
- **Concerns:**
  - **Blocker:** No audio component — cannot be used for speech-to-token fine-tuning
  - Gated access requires contact information

### 4. IIITH Indic Speech
- **Source:** https://huggingface.co/datasets/iitm-ddp/iiith-indic-speech
- **Language:** Indic languages including Marathi
- **Task:** Speech corpus (ASR-focused)
- **Audio:** Audio folder format, speech samples
- **Transcript:** Not clearly documented
- **Speakers:** Unknown
- **Size:** 9,637 total samples | 2.78 GB | multiple languages combined
- **Splits:** Not documented
- **License:** Not documented
- **Preprocessing:** Audio folder format (requires extraction)
- **Suitability:**
  - ⚠️ Marathi is included but not isolated in documentation
  - ⚠️ No dataset card provided (metadata unavailable)
  - ⚠️ Total size includes multiple languages (Marathi share unknown)
  - ⚠️ No license information
  - ⚠️ Preprocessing requirements unclear
- **Concerns:**
  - Lack of documentation makes assessment difficult
  - Unknown whether Marathi-specific samples are substantial
  - No clarity on data organization or labeling

### 5. Text-Speech Indic (almlengineer143)
- **Source:** https://huggingface.co/datasets/almlengineer143/text_speech_indic
- **Language:** Indic languages (Marathi component unknown)
- **Task:** Text-to-Speech
- **Audio:** Parquet format
- **Transcript:** Text transcripts
- **Speakers:** Unknown
- **Size:** 312 samples | 291 MB | very small
- **Splits:** Default split only
- **License:** Not documented
- **Preprocessing:** None documented
- **Suitability:**
  - ❌ **Too small** for meaningful training (312 total samples)
  - ⚠️ No documentation on Marathi coverage
  - ⚠️ No README or licensing information
- **Concerns:**
  - Insufficient sample size
  - No metadata or documentation
  - Unclear dataset composition

### 6. Equal AI UltraVox Indic Speech
- **Source:** https://huggingface.co/datasets/equal-ai/ultravox-indic-speech
- **Language:** English and Hindi only (❌ No Marathi)
- **Task:** Speech Recognition and TTS
- **Audio:** Large multilingual speech dataset
- **Transcript:** Cleaned transcripts
- **Speakers:** Multiple
- **Size:** 4.2M samples | 1.19 TB
- **Splits:** Train/validation splits documented
- **License:** Not explicitly clear
- **Preprocessing:** Complex (large dataset)
- **Suitability:**
  - ❌ **No Marathi coverage** — English + Hindi only
  - ❌ Gated access (requires contact information)
  - ❌ Extremely large (overkill for assignment scope)
- **Concerns:**
  - Explicitly excludes Marathi
  - Not viable for this assignment

---

## Comparison Table

| Dataset | Marathi? | Audio+Text | Samples | Size | License | Access | Suitability |
|---------|----------|-----------|---------|------|---------|--------|-------------|
| **Shunya Labs** | ✅ Native | ✅ Yes | 352K | 43.7GB | ⚠️ Unclear | ✅ Open | ⭐⭐⭐⭐ Promising |
| **FLEURS mr_in** | ✅ Included | ✅ Yes | 3.3K train | 12h audio | ✅ CC-BY 4.0 | ✅ Open | ⭐⭐⭐⭐ Good |
| **PAARI Marathi** | ✅ Native | ❌ Text only | N/A | 21.5GB | ⚠️ Gated | ⚠️ Gated | ❌ Not viable |
| **IIITH Indic** | ⚠️ Included | ✅ Yes | 9.6K total | 2.78GB | ⚠️ Unknown | ✅ Open | ⭐⭐ Unclear |
| **Text-Speech Indic** | ⚠️ Possibly | ✅ Yes | 312 | 291MB | ⚠️ None | ✅ Open | ❌ Too small |
| **UltraVox Indic** | ❌ No | ✅ Yes | 4.2M | 1.19TB | ⚠️ Unclear | ⚠️ Gated | ❌ No Marathi |

---

## Selected Dataset

**Primary: Google FLEURS (Marathi: mr_in)**

**Rationale:**

1. **Proven Quality & Documentation**
   - Published academic dataset (Google Research, 2022)
   - Peer-reviewed paper with clear methodology
   - Explicit licensing (CC-BY 4.0) permits research fine-tuning
   - Reproducible data collection with validation pipeline

2. **Clean Audio/Text Pairing**
   - Audio samples directly loadable via Hugging Face `datasets` library
   - High-quality manual transcriptions with multiple normalization variants
   - Multiple speakers (gender-balanced) reduce overfitting risk
   - Text cleaned and normalized consistently

3. **Practical Feasibility**
   - Small enough to download/process on CPU development machine
   - 16 kHz → 24 kHz upsampling is straightforward
   - ~4.7K total samples provide meaningful training diversity
   - ~12 hours of audio is realistic for fine-tuning on Bodhan base model

4. **Marathi-Specific Coverage**
   - Part of South-Asia language group (Marathi is "unseen" in pre-training for mSLAM baseline)
   - Consistent Devanagari script handling
   - Parallel sentences maintain semantic consistency across recordings
   - Speaker separation between train/test prevents overfitting

5. **Reduced Preprocessing Burden**
   - No authentication or gated access barriers
   - Parquet format with built-in train/val/test splits
   - Text preprocessing already complete (normalization options provided)
   - Standard Python/Hugging Face workflow

---

### Secondary: Shunya Labs Marathi Speech Dataset

**Why Secondary:**
- Significantly larger (352K samples) provides more training data if needed
- Native Marathi corpus (not multilingual ensemble)
- Worth investigating after initial FLEURS-based fine-tuning

**Blocker for Now:**
- Licensing terms not explicitly documented (risk of future misuse claims)
- No published methodology or data quality assurance
- Metadata sparse (speaker info, quality metrics, domain breakdown)
- Would require more extensive preprocessing inspection before use

---

## What Remains Unknown

Until the selected dataset(s) are downloaded and inspected:

1. **FLEURS Marathi Audio Quality**
   - Actual audio bitrate and compression format
   - Presence of background noise, room acoustics, or artifacts
   - Utterance duration distribution and statistics
   - Speaker consistency and voice characteristics

2. **Shunya Labs Marathi Dataset Details**
   - Actual Marathi language coverage (exact sample count)
   - Audio sampling rate and format metadata
   - Text preprocessing applied (punctuation, normalization, script validation)
   - Domain breakdown (news, conversational, read speech, etc.)
   - Whether dataset includes quality control metadata
   - Actual licensing terms (may be clear in README upon download)

3. **Compatibility with Bodhan Indic-Speak**
   - Whether Marathi phonetics align with trained Bodhan model expectations
   - Any script encoding issues (UTF-8 Devanagari consistency)
   - Speaker prosody/accent characteristics (Bodhan may have specific expectations)

4. **SNAC Serialization in Practice**
   - Whether FLEURS/Shunya Labs audio properties align with validated SNAC pipeline
   - Real-world edge cases (silence, OOV phonemes, speaker variation)
   - Practical token distribution in actual training samples

---

## Next Steps (When Fine-Tuning Begins)

1. **Download FLEURS mr_in** to inspect audio samples and transcription quality
2. **Resample to 24 kHz** using validated pipeline (librosa or similar)
3. **Validate SNAC serialization** on real samples (use existing validation script)
4. **If FLEURS insufficient:** Download Shunya Labs dataset and inspect licensing/metadata
5. **Create data loader** for Bodhan fine-tuning with proper batching and preprocessing

---

## References

- **FLEURS Dataset Paper:** Conneau et al. (2022) — https://arxiv.org/abs/2205.12446
- **Hugging Face FLEURS:** https://huggingface.co/datasets/google/fleurs
- **Shunya Labs Marathi Speech:** https://huggingface.co/datasets/shunyalabs/marathi-speech-dataset
- **Bodhan Indic-Speak Model:** https://huggingface.co/bodhan-ai/indic-speak (model reference only)
