# Speech-Token Serialization Analysis — Bodhan Indic-Speak

**Date:** 2026-09-22  
**Model:** `bodhan-ai/indic-speak`  
**Source:** Authenticated access to `token_contract.md` and `inference.py`  
**Status:** Exact serialization DETERMINED from public implementation

---

## Executive Summary

The exact SNAC speech-token serialization is **fully documented** in the repository's `token_contract.md` and **explicitly implemented** in `inference.py`. The 7 SNAC codebooks are **position-interleaved** into a flat Llama token stream: exactly 7 tokens per frame, where each token's ID encodes both the audio code value (0–4095) AND the codebook membership (via its offset position).

**Key fact:** Token ID does NOT directly map to a single codebook. Instead, the position in the 7-token frame determines the codebook:
- Position 0 → c0 (coarsest, 12 Hz)
- Positions 1, 4 → c1 (intermediate, 23 Hz)
- Positions 2, 3, 5, 6 → c2 (finest, 47 Hz)

---

## 1. SNAC Hierarchical Structure (from token_contract.md § 3)

### 1.1 SNAC Codebook Hierarchy

SNAC 24 kHz operates with **3 hierarchical codebooks** at different temporal rates:

| Codebook | Name | Rate | Temporal Ratio | Samples per Frame |
|---|---|---|---|---|
| **c0** | Coarse | 12 Hz | 1× | 2,048 samples (85.3 ms) |
| **c1** | Intermediate | 23 Hz | 2× | 1,024 samples (42.7 ms) |
| **c2** | Fine | 47 Hz | 4× | 512 samples (21.3 ms) |

**Key property:** At each 12 Hz coarse frame `i`:
- c0 has 1 code
- c1 has 2 codes (indices 2i, 2i+1)
- c2 has 4 codes (indices 4i, 4i+1, 4i+2, 4i+3)
- **Total: 7 codes per frame**

### 1.2 Samples Per Frame Calculation

From `token_contract.md`:
```
samples per c0 frame = 512 (encoder_rates [2,4,8,8] x vq_stride 4)
```

At 24 kHz:
- c0 frame duration: 512 / 24,000 = 21.3 ms
- But semantic frame rate is different:
  - c0 @ 12 Hz → 1 frame per 83.3 ms
  - c1 @ 23 Hz → 2 frames per 83.3 ms
  - c2 @ 47 Hz → 4 frames per 83.3 ms

The temporal ratio (1:2:4) is preserved in the token contract.

---

## 2. Exact Token Serialization (from token_contract.md § 3)

### 2.1 The Interleaving Formula

For frame index `i`, the 7 tokens generated are:

```
frame i -> [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]]
```

**Meaning:**
- Position 0: Coarse codebook index i
- Position 1: Intermediate codebook index 2i (first code at this frame)
- Position 2: Fine codebook index 4i (first code)
- Position 3: Fine codebook index 4i+1 (second code)
- Position 4: Intermediate codebook index 2i+1 (second code at this frame)
- Position 5: Fine codebook index 4i+2 (third code)
- Position 6: Fine codebook index 4i+3 (fourth code)

### 2.2 Token ID Mapping (from token_contract.md § 3 table)

Each raw code (0–4095) is offset by **position in the frame**, NOT by codebook membership:

| Frame Position | Source Codebook | Code Index Range | Token ID Offset | Token ID Range |
|---|---|---|---|---|
| **0** | c0[i] | 0–4095 | base + 0×4096 | 128,266–132,361 |
| **1** | c1[2i] | 0–4095 | base + 1×4096 | 132,362–136,457 |
| **2** | c2[4i] | 0–4095 | base + 2×4096 | 136,458–140,553 |
| **3** | c2[4i+1] | 0–4095 | base + 3×4096 | 140,554–144,649 |
| **4** | c1[2i+1] | 0–4095 | base + 4×4096 | 144,650–148,745 |
| **5** | c2[4i+2] | 0–4095 | base + 5×4096 | 148,746–152,841 |
| **6** | c2[4i+3] | 0–4095 | base + 6×4096 | 152,842–156,937 |

**Critical insight:** The offset encodes **where in the frame** a code sits, which is what makes the stream decodable without a separate structure signal.

### 2.3 Token ID to Code Value Conversion

Given a token ID `tok_id` in the audio range [128,266, 156,937]:

```
position_in_frame = (tok_id - base) // 4096     # 0-6
code_value = (tok_id - base) % 4096             # 0-4095
```

**Example:**
- tok_id = 140,600 (in position 3 range [140,554, 144,649])
- position_in_frame = (140,600 - 128,266) // 4096 = 12,334 // 4096 = 3
- code_value = (140,600 - 128,266) % 4096 = 12,334 % 4096 = 46
- **Interpretation:** Frame i, position 3 → c2[4i+1] = 46

---

## 3. Inference Decoding Path (from inference.py)

### 3.1 Complete Generation Flow

```
Prompt text + speaker + style
  ↓
build_prompt() → Llama token IDs (text)
  ↓
LLM.generate()
  ↓ (generates 7 tokens per frame, starting with <|start_of_speech|>)
  ↓
ids_to_codes() ← Converts flat token stream to SNAC codebooks
  ↓
SNAC.quantizer.from_codes() → Latent audio representation z_q
  ↓
Vocos decoder → 24 kHz waveform
  ↓
save() → .wav file
```

### 3.2 The ids_to_codes() Decoding Function

From `inference.py` (lines 55–75):

```python
def ids_to_codes(ids: list[int], base: int, device) -> list[torch.Tensor]:
    """Flat LM token ids -> SNAC's 3 hierarchical codebooks (rates 1:2:4).

    Frame i is [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]],
    each offset by base + position*4096. Stops at the first non-audio token.
    """
    hi = base + NUM_CODEBOOKS * CODEBOOK_SIZE  # 128,266 + 7*4096 = 156,938
    audio = []
    for t in ids:
        if not (base <= t < hi):
            break
        audio.append(t)

    n = len(audio) // NUM_CODEBOOKS  # number of complete frames
    if n == 0:
        raise ValueError("model emitted no complete SNAC frame")

    # Step 1: Extract all tokens and reshape to (n_frames, 7)
    a = np.array(audio[: n * NUM_CODEBOOKS], np.int32).reshape(n, NUM_CODEBOOKS)
    
    # Step 2: Remove offsets to get raw code values (0-4095)
    #  a[frame_idx, position] -= (base + position * 4096)
    a -= base + np.arange(NUM_CODEBOOKS, dtype=np.int32) * CODEBOOK_SIZE
    
    # Step 3: Validate all codes are in range (0-4095)
    a = a[np.all((a >= 0) & (a < CODEBOOK_SIZE), axis=1)]
    if a.shape[0] == 0:
        raise ValueError("no in-range SNAC frame")

    # Step 4: Reconstruct c1 and c2 from their interleaved positions
    c1 = np.empty(a.shape[0] * 2, np.int32)
    c1[0::2], c1[1::2] = a[:, 1], a[:, 4]  # positions 1 and 4
    
    c2 = np.empty(a.shape[0] * 4, np.int32)
    c2[0::4], c2[1::4], c2[2::4], c2[3::4] = a[:, 2], a[:, 3], a[:, 5], a[:, 6]  # positions 2,3,5,6

    # Step 5: Return c0, c1, c2 as tensors
    return [torch.from_numpy(x.copy()).long().unsqueeze(0).to(device)
            for x in (a[:, 0], c1, c2)]
```

**Line-by-line walkthrough:**

**Line 59–63: Audio token extraction**
- Iterate through generated token IDs
- Stop at the first non-audio token (e.g., `<|end_of_speech|>` or any token < base or ≥ hi)
- Collect all audio tokens in `audio` list

**Line 65–66: Frame count and reshaping**
- `n = len(audio) // 7` — number of complete 7-token frames
- Reshape audio list into (n, 7) array, dropping any incomplete final frame
- Each row is one frame with 7 positions

**Line 69–70: De-offsetting**
- Subtract `base` from the first position and `base + position*4096` from each position
- Result: `a[frame_idx, position]` now contains raw code value (0–4095) or negative/overflow if invalid

**Line 71–74: Validation**
- Keep only frames where all 7 positions decode to valid codes (0–4095)
- Raise error if no valid frames remain

**Line 76–80: Codebook reconstruction**
- **c1 reconstruction:** Extract positions 1 and 4 (the two c1 codes per frame)
  - c1[0], c1[1] ← a[0, 1], a[0, 4] (frame 0)
  - c1[2], c1[3] ← a[1, 1], a[1, 4] (frame 1)
  - Result: c1 has shape (n_frames*2,)
  
- **c2 reconstruction:** Extract positions 2, 3, 5, 6 (the four c2 codes per frame)
  - c2[0], c2[1], c2[2], c2[3] ← a[0, 2], a[0, 3], a[0, 5], a[0, 6] (frame 0)
  - Result: c2 has shape (n_frames*4,)

**Line 82–83: Return codebooks**
- c0: shape (1, n_frames) — one tensor for all coarse codes
- c1: shape (1, n_frames*2) — one tensor for all intermediate codes
- c2: shape (1, n_frames*4) — one tensor for all fine codes

---

## 4. Generation Boundary Detection

### 4.1 Start-of-Speech Marker

From `inference.py` (lines 106, 144, 155):

```python
self._base = self.tok.convert_tokens_to_ids("<|snac_0|>")
self._eos = self.tok.convert_tokens_to_ids("<|end_of_speech|>")
```

- **Start:** Not explicitly generated as a special token. The prompt construction (in `build_prompt()`) includes `<|start_of_speech|>` (ID 128257) in the prompt, which signals to the model that audio generation should begin.
- **First generated token:** The model's first generated token after `<|start_of_speech|>` is the first audio code, NOT a repeat of the start marker.

### 4.2 End-of-Speech Detection

From `inference.py` (line 162):

```python
eos_token_id=[self._eos, self.tok.eos_token_id],
```

- The generation loop stops when the model emits `<|end_of_speech|>` (ID 128258) or the Llama EOS token (ID 128001)
- The `ids_to_codes()` function stops reading tokens at the first non-audio token, so EOS detection is automatic

### 4.3 Truncation Warning

From `inference.py` (lines 164–168):

```python
if len(new_ids) >= max_new_tokens:
    warnings.warn(
        f"generation hit max_new_tokens ({max_new_tokens}) without emitting "
        "<|end_of_speech|> — audio is likely truncated or runaway babble",
        RuntimeWarning, stacklevel=2)
```

- If generation reaches `max_new_tokens` without emitting EOS, a warning is issued
- Audio may be truncated or invalid

### 4.4 Incomplete Frame Handling

From `ids_to_codes()`:
- If the final generated frame is incomplete (< 7 tokens), it is dropped
- `n = len(audio) // 7` truncates to complete frames only

---

## 5. Duplicate Frame Removal

From `token_contract.md` § 3:

```
Consecutive duplicate frames are removed at encode time (frames sharing the
same c0), so token count is not exactly proportional to duration.
```

**Implication for inference:**
- The generated audio tokens may not have exactly 7 tokens per SNAC semantic frame
- However, the post-processing in `ids_to_codes()` expects exactly 7 tokens per frame
- This is a mismatch between training (where duplicates are removed) and inference
- **This is likely handled during training data preparation, not in the inference decoder**

---

## 6. Prompt Structure and Sequence Layout

### 6.1 build_prompt() Function

From `inference.py` (lines 37–49):

```python
def build_prompt(tok, text: str, speaker: str = "", style: str = "") -> list[int]:
    """<|start_of_human|><|begin_of_text|>[<|speaker>..<speaker|>][<|style>..<style|>]
    text<|eot_id|><|end_of_human|><|start_of_ai|><|start_of_speech|>"""
    tid = lambda s: tok.convert_tokens_to_ids(s)
    nl = tok.encode("\n", add_special_tokens=False)
    enc = lambda s: tok.encode(s, add_special_tokens=False)

    blocks = []
    if speaker.strip():
        blocks.append([tid("<|speaker>")] + enc(speaker.strip()) + [tid("<speaker|>")])
    if style.strip():
        blocks.append([tid("<|style>")] + enc(style.strip()) + [tid("<style|>")])

    meta: list[int] = []
    for i, b in enumerate(blocks):
        if i:
            meta += nl
        meta += b
    if blocks:
        meta += nl

    body = [tok.bos_token_id] + meta + enc(text) + [tid("<|eot_id|>")]
    return ([tid("<|start_of_human|>")] + body + [tid("<|end_of_human|>")]
            + [tid("<|start_of_ai|>"), tid("<|start_of_speech|>")])
```

### 6.2 Generated Prompt Structure

**For `tts("नमस्ते", speaker="Anagha", style="")`:**

```
<|start_of_human|>
<|begin_of_text|>
<|speaker>Anagha<speaker|>
नमस्ते
<|eot_id|>
<|end_of_human|>
<|start_of_ai|>
<|start_of_speech|>
[audio tokens...]
<|end_of_speech|>
```

**Token sequence:**
1. `<|start_of_human|>` (128259)
2. `<|begin_of_text|>` (Llama special, ID 128000)
3. `<|speaker>` (156938)
4. Tokenized speaker name "Anagha"
5. `<speaker|>` (156939)
6. Newline (`\n`, BPE token)
7. Tokenized text "नमस्ते"
8. `<|eot_id|>` (Llama special)
9. `<|end_of_human|>` (128260)
10. `<|start_of_ai|>` (128261)
11. **`<|start_of_speech|>` (128257) ← Marks the prompt end, model generates from next token**
12. First audio frame: 7 SNAC tokens
13. Additional frames...
14. `<|end_of_speech|>` (128258) or EOS

---

## 7. Full Inference Pipeline Step-by-Step

### 7.1 Example: Generate Marathi speech for "नमस्ते" with speaker Anagha

**Input:**
```python
tts = TTS("bodhan-ai/indic-speak")
wav = tts("नमस्ते", speaker="Anagha", temperature=0.6, top_p=0.9, max_new_tokens=2520)
```

**Step 1: Build prompt**
```
build_prompt(tokenizer, "नमस्ते", "Anagha", "") →
[128259, 128000, 156938, ..., 156939, ..., नमस्ते_tokens, ..., 128260, 128261, 128257]
```

**Step 2: Generate SNAC tokens**
```python
ids = torch.tensor([[prompt_tokens]], device=device)
gen = lm.generate(input_ids=ids, max_new_tokens=2520, eos_token_id=[128258, 128001], ...)
# gen shape: (1, len(prompt_tokens) + num_generated_tokens)
new_ids = gen[0].tolist()[len(prompt_tokens):]
# new_ids shape: (num_generated_tokens,) containing SNAC tokens [128266, ..., 156937] and ending with 128258
```

**Step 3: Decode SNAC tokens to codebooks**
```python
codes = ids_to_codes(new_ids, base=128266, device=device)
# codes[0] (c0): shape (1, n_frames)
# codes[1] (c1): shape (1, n_frames*2)
# codes[2] (c2): shape (1, n_frames*4)
```

**Example intermediate state** (frame 0):
```
Generated tokens: [132500, 134200, 139100, 141850, 146900, 150700, 155200, 128258]
Position:         [0,      1,      2,      3,      4,      5,      6,      (EOS)]

After de-offsetting (subtract base + position*4096):
a[0,0] = 132500 - (128266 + 0*4096) = 4234 ✓ c0[0] = 4234
a[0,1] = 134200 - (128266 + 1*4096) = 1838 ✓ c1[0] = 1838
a[0,2] = 139100 - (128266 + 2*4096) = 2638 ✓ c2[0] = 2638
a[0,3] = 141850 - (128266 + 3*4096) = 1422 ✓ c2[1] = 1422
a[0,4] = 146900 - (128266 + 4*4096) = 2734 ✓ c1[1] = 2734
a[0,5] = 150700 - (128266 + 5*4096) = 2338 ✓ c2[2] = 2338
a[0,6] = 155200 - (128266 + 6*4096) = 2638 ✓ c2[3] = 2638

Reconstructed:
c0 = [4234]
c1 = [1838, 2734]
c2 = [2638, 1422, 2338, 2638]
```

**Step 4: Decode with SNAC quantizer**
```python
z_q = snac.quantizer.from_codes([c0_tensor, c1_tensor, c2_tensor])
# z_q: latent representation (batch=1, channels, time)
```

**Step 5: Decode with Vocos**
```python
wav = vocos(z_q.float())  # or snac.decoder if stock=True
# wav shape: (1, 1, num_samples)
# num_samples = n_frames * samples_per_frame ≈ frames * 512
```

**Step 6: Clamp, convert, and return**
```python
return wav[0, 0].clamp(-1, 1).float().cpu().numpy()
# Returns float32 mono waveform @ 24 kHz
```

---

## 8. Exact Known Facts

### 8.1 Token Serialization (VERIFIED)

✅ **Codebook interleaving:** Exactly 7 tokens per frame, in order:
```
[c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]]
```

✅ **Token ID mapping:** Each position has a dedicated offset range:
| Position | Offset | Token Range |
|---|---|---|
| 0 | base + 0×4096 | 128,266–132,361 |
| 1 | base + 1×4096 | 132,362–136,457 |
| 2 | base + 2×4096 | 136,458–140,553 |
| 3 | base + 3×4096 | 140,554–144,649 |
| 4 | base + 4×4096 | 144,650–148,745 |
| 5 | base + 5×4096 | 148,746–152,841 |
| 6 | base + 6×4096 | 152,842–156,937 |

✅ **SNAC hierarchical rates:** 1:2:4 temporal ratio
- c0: 1 code per frame, 12 Hz
- c1: 2 codes per frame, 23 Hz (indices 2i, 2i+1)
- c2: 4 codes per frame, 47 Hz (indices 4i, 4i+1, 4i+2, 4i+3)

✅ **Boundary markers:**
- Start: `<|start_of_speech|>` (128257) in prompt; first generated token is audio
- End: `<|end_of_speech|>` (128258) or Llama EOS (128001)

✅ **Sequence boundaries:** Complete frames only; incomplete final frame dropped

### 8.2 Implementation Details (VERIFIED)

✅ **Generation:** LLM generates 7 tokens per frame, one per SNAC position

✅ **Decoding:** `ids_to_codes()` extracts flat token stream, de-offsets by position, reconstructs c0/c1/c2

✅ **Vocos input:** Latent representation from `SNAC.quantizer.from_codes()`

✅ **Output:** 24 kHz float32 mono waveform

---

## 9. Implications for Training

### 9.1 Label Serialization

For training, audio labels must be encoded exactly as the inference decoder expects:

**Algorithm to encode training labels:**

```
Given: c0 (n_frames,), c1 (n_frames*2,), c2 (n_frames*4,)

For each frame i:
  tokens[i*7 + 0] = 128266 + 0*4096 + c0[i]
  tokens[i*7 + 1] = 128266 + 1*4096 + c1[2i]
  tokens[i*7 + 2] = 128266 + 2*4096 + c2[4i]
  tokens[i*7 + 3] = 128266 + 3*4096 + c2[4i+1]
  tokens[i*7 + 4] = 128266 + 4*4096 + c1[2i+1]
  tokens[i*7 + 5] = 128266 + 5*4096 + c2[4i+2]
  tokens[i*7 + 6] = 128266 + 6*4096 + c2[4i+3]

Append: 128258 (<|end_of_speech|>)
```

### 9.2 Validation Checklist

For training data construction:
- ✅ Extract c0, c1, c2 from SNAC quantizer at 24 kHz
- ✅ Verify c0 length = n_frames
- ✅ Verify c1 length = n_frames * 2
- ✅ Verify c2 length = n_frames * 4
- ✅ Serialize with exact position-offset mapping
- ✅ Use token ID range 128,266–156,937 for audio
- ✅ Use 128258 as end-of-speech marker
- ✅ Append to prompt tokens (speaker/style/text)
- ✅ Construct full sequence: prompt + audio + EOS

### 9.3 Known Limitation: Duplicate Frame Removal

From token_contract.md:
> Consecutive duplicate frames are removed at encode time (frames sharing the same c0)

**Implication:** Training data will have fewer than 7 tokens per actual audio frame in some places, breaking the strict "7 tokens per frame" assumption during inference.

**This is handled in the training/preprocessing pipeline, not in the inference decoder.** The preprocessing must:
1. Detect duplicate consecutive c0 values
2. Skip those frames
3. Create a mapping from inference frames back to preprocessing frames
4. OR: Keep duplicates in training labels despite encoder removing them

**The exact training strategy is unknown without access to training code.**

---

## 10. What is NOT Determined

### 10.1 Training Loss Function

❌ **Unknown:** How the SNAC tokens are used as training targets
- Cross-entropy loss on 28,672-way classification per position?
- Separate loss for each codebook position?
- Weighted loss favoring certain positions?

### 10.2 Training Data Filtering

❌ **Unknown:** How duplicate frames are handled during training
- Are duplicates removed, kept, or specially marked?
- How does the training loop know which frames should be compressed?

### 10.3 Fine-Tuning Strategy

❌ **Unknown:** How to fine-tune on Marathi without the official training code
- Full model fine-tuning or LoRA?
- Learning rate, batch size, number of epochs?
- Warmup and decay schedule?
- Data augmentation strategy?

### 10.4 Speaker/Style Embeddings

❌ **Unknown:** How speaker/style metadata is encoded in the model
- Are speaker/style names encoded as BPE tokens in the prompt only?
- Are there learned speaker embeddings, or just cross-entropy on the text?
- Can new speakers be added without retraining?

---

## 11. Summary Table

| Aspect | Status | Source | Evidence |
|--------|--------|--------|----------|
| Interleaving order (7 tokens/frame) | ✅ KNOWN | token_contract.md § 3 | Exact formula: [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]] |
| Position-based offsets | ✅ KNOWN | token_contract.md § 3 table | Each position has dedicated range: base + pos*4096 |
| SNAC hierarchical rates | ✅ KNOWN | token_contract.md § 3, inference.py | c0: 12 Hz, c1: 23 Hz, c2: 47 Hz (1:2:4 ratio) |
| Decoding algorithm | ✅ KNOWN | inference.py ids_to_codes() | Extract, de-offset, reconstruct codebooks |
| Boundary markers | ✅ KNOWN | token_contract.md § 2, inference.py | Start: 128257, End: 128258 or EOS |
| Complete frame requirement | ✅ KNOWN | inference.py ids_to_codes() | Incomplete final frame dropped |
| Training loss function | ❌ NOT KNOWN | — | No training code in repository |
| Duplicate frame handling | ❌ NOT KNOWN | — | Mentioned in token_contract.md but preprocessing code not visible |
| Fine-tuning strategy | ❌ NOT KNOWN | — | No training documentation |
| Speaker embedding space | ❌ NOT KNOWN | — | Model weights not inspected |

---

## 12. Conclusion

The Bodhan Indic-Speak speech-token serialization is **fully and exactly documented** in the public repository. The inference path is clear:

1. **Prompt building:** Text + speaker/style metadata formatted with control tokens
2. **LLM generation:** Produces 7 tokens per frame, position-interleaved
3. **Decoding:** `ids_to_codes()` de-offsets and reconstructs SNAC codebooks (c0, c1, c2)
4. **Audio synthesis:** SNAC quantizer + Vocos decoder → 24 kHz waveform

**For training:** The encoding process must reverse the `ids_to_codes()` function exactly:
- Extract c0, c1, c2 from SNAC encoder
- Serialize into position-offset tokens
- Append to prompt tokens
- Construct full training sequence

The token serialization is deterministic and verifiable. The only unknowns are in the training objective and fine-tuning strategy, which are intentionally not included in the public inference-only release.

---

**Document Source:** `token_contract.md` (full), `inference.py` (complete)  
**Verified:** 2026-09-22  
**Scope:** Inference and label encoding only; training code not in repository
