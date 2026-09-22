# Training Formulation Audit Report

**Date:** 2026-09-22  
**Status:** Pre-training review  
**Objective:** Verify training.py and config.py against authenticated Bodhan inspection  

---

## 1. Audit Findings

### 1.1 Token ID Constants ✅ VERIFIED

**Source:** token_contract.md  
**Implementation:** config.py

| Token | Expected ID | Actual ID | Status |
|-------|------------|-----------|--------|
| `<\|start_of_human\|>` | 128,259 | 128259 | ✅ |
| `<\|end_of_human\|>` | 128,260 | 128260 | ✅ |
| `<\|start_of_ai\|>` | 128,261 | 128261 | ✅ |
| `<\|end_of_ai\|>` | 128,262 | 128262 | ✅ |
| `<\|start_of_speech\|>` | 128,257 | 128257 | ✅ |
| `<\|end_of_speech\|>` | 128,258 | 128258 | ✅ |
| SNAC start | 128,266 | 128266 | ✅ |
| SNAC end | 156,937 | 156937 | ✅ |

**Conclusion:** Token constants are correct and match token_contract.md exactly.

---

### 1.2 SNAC Codebook Layout ✅ VERIFIED

**Source:** token_contract.md: "7 codebooks × 4,096 codes = 28,672 tokens"  
**Implementation:** config.py

```python
BODHAN_NUM_CODEBOOKS = 7
BODHAN_CODEBOOK_SIZE = 4096
BODHAN_SNAC_COUNT = 28672
```

**Actual range:** 128,266–156,937  
**Count:** 156,937 - 128,266 + 1 = 28,672 ✅

**Conclusion:** SNAC token allocation is correct.

---

### 1.3 Prompt Construction — CRITICAL ISSUE FOUND ⚠️

**Source:** inference.py (authenticated inspection)  
**Expected Prompt Structure:**
```
<|start_of_human|>
<|begin_of_text|>
[<|speaker>Speaker Name<speaker|>]
[<|style>Style<style|>]
{text}
<|eot_id|>
<|end_of_human|>
<|start_of_ai|>
<|start_of_speech|>         ← (model generates FROM here, not in prompt)
```

**Key Point from Inspection:** 
> "<|start_of_speech|> is the model's first GENERATED token in the audio span, not a prompt prefix token."

**Current Implementation (training.py, line ~45-60):**
```python
prompt_parts.extend([
    text,
    "<|eot_id|>",
    "<|end_of_human|>",
    "<|start_of_ai|>",
    "<|start_of_speech|>",  # ⚠️ INCLUDED IN PROMPT STRING
])
```

**Issue:** The prompt string includes `<|start_of_speech|>` as if it's part of the context. But according to the authenticated inspection, this token should be the FIRST GENERATED token, not part of the prompt context.

**Impact on Training:**
- When tokenized, `<|start_of_speech|>` becomes a token ID in the prompt
- Then the code adds `[BODHAN_START_OF_SPEECH]` again (line ~240):
  ```python
  input_ids = (
      prompt_token_ids +          # May or may not include start_of_speech
      [BODHAN_START_OF_SPEECH] +  # Explicitly add it again
      speech_tokens.tolist() +
      [BODHAN_END_OF_SPEECH]
  )
  ```
- This creates ambiguity about whether start_of_speech is tokenized from the string or added explicitly

**Resolution:** Prompt structure should be:
1. Build prompt ending with `<|start_of_ai|>` (NOT including start_of_speech)
2. Tokenize prompt
3. Explicitly construct: `prompt_tokens + [start_of_speech] + speech_tokens + [end_of_speech]`
4. Labels should supervise start_of_speech (not mask it), since the model must learn to generate it

---

### 1.4 Label Masking Strategy — ISSUE FOUND ⚠️

**Expected (from authenticated inspection and standard causal LM):**
```
Prompt context:          <|start_of_human|> ... <|start_of_ai|>
Generated sequence:      <|start_of_speech|> [SNAC codes...] <|end_of_speech|>

input_ids:  [prompt_tokens...] [start_of_speech] [snac_tokens...] [end_of_speech]
labels:     [    -100     ...] [start_of_speech] [snac_tokens...] [end_of_speech]
```

**Current Implementation (training.py, line ~246-250):**
```python
labels = (
    [-100] * (text_length + 1) +  # Text + start_of_speech BOTH MASKED
    speech_tokens.tolist() +       # Speech tokens supervised
    [BODHAN_END_OF_SPEECH]         # End token supervised
)
```

**Issue:** Masks `start_of_speech` with -100, meaning the model is NOT trained to generate this token. But from the authenticated inspection, the model must learn to emit `<|start_of_speech|>` when it sees the prompt ending with `<|start_of_ai|>`.

**Correction Needed:**
```python
labels = (
    [-100] * text_length +         # Only text is masked
    [BODHAN_START_OF_SPEECH] +     # start_of_speech MUST be supervised
    speech_tokens.tolist() +       # Speech tokens supervised
    [BODHAN_END_OF_SPEECH]         # End token supervised
)
```

---

### 1.5 Speaker/Style Encoding ✅ VERIFIED

**Source:** inference.py: `<|speaker>name<speaker|>` and `<|style>text<style|>`  
**Implementation:** training.py, lines 65-70

```python
if speaker:
    prompt_parts.append(f"<|speaker>{speaker}<speaker|>")
if style:
    prompt_parts.append(f"<|style>{style}<style|>")
```

**Verification:** Matches inference.py exactly. Speaker and style are text-encoded blocks, not special tokens.  
**Status:** ✅ CORRECT

**Default values:** Anagha (verified from voices.md) and NEUTRAL (verified from inference.py examples)  
**Status:** ✅ CORRECT

---

### 1.6 Causal LM Loss Setup ✅ GENERALLY CORRECT

**Expected:**
- Only positions where `labels != -100` contribute to loss
- This is standard transformers training (e.g., HF Trainer)

**Implementation:** 
- TrainingCollator initializes labels with -100 (line 311)
- Fills in actual labels for non-padded positions (line 319)
- Attention mask separates real from padding (line 320)

**Status:** ✅ Correct structure (but needs fix to label masking strategy above)

---

### 1.7 Attention Mask ✅ VERIFIED

**Expected:** 
- 1 for real tokens
- 0 for padding

**Implementation:** TrainingCollator, line 320
```python
attention_mask[i, :seq_len] = 1  # Real tokens
# (padding already initialized to 0)
```

**Status:** ✅ CORRECT

---

### 1.8 Sequence Length and Constraints ✅ VERIFIED

**Source:** tokenizer_config.json: `model_max_length: 131072`  
**Implementation:** config.py

```python
BODHAN_MODEL_MAX_LENGTH = 131072
max_seq_length = 2600
```

**Validation:** `__post_init__` checks `max_seq_length <= BODHAN_MODEL_MAX_LENGTH`  
**Status:** ✅ CORRECT (2,600 << 131,072)

---

### 1.9 Padding and Truncation ✅ VERIFIED

**Implementation:** TrainingCollator
- Uses `actual_max_len = min(max(example lengths), max_length)`
- Pads shorter examples to this length
- Sets `labels[padding] = -100`
- Sets `attention_mask[padding] = 0`

**Status:** ✅ CORRECT

---

## 2. Summary of Issues

### Critical Issues Requiring Fix

| Issue | Location | Severity | Fix Required |
|-------|----------|----------|--------------|
| Prompt includes `<\|start_of_speech\|>` | training.py:build_prompt() | CRITICAL | Remove from prompt; keep explicit addition |
| Label masking masks `start_of_speech` | training.py:__getitem__() | CRITICAL | Supervise instead of mask start_of_speech token |

### Verified & Correct

✅ All token ID constants  
✅ SNAC codebook layout  
✅ Speaker/style text encoding  
✅ Causal LM structure  
✅ Attention masks  
✅ Sequence length constraints  
✅ Padding/truncation logic  

---

## 3. Root Cause Analysis

### Why These Issues Exist

**Prompt Issue:** The build_prompt function was designed to return a complete prompt string (including start_of_speech), but this conflicts with the inference.py architecture where start_of_speech is the FIRST GENERATED token, not part of the prompt context.

**Label Masking Issue:** Followed from the prompt issue—if start_of_speech was in the prompt, it seemed logical to mask it. But since it should be generated, it must be supervised in the labels.

### Verification from Dry-Run

The dry-run succeeded with:
- text_length = 48 (prompt tokens)
- supervised positions = 470 (speech tokens only)
- ignored positions = 49 (text + what was supposed to be start_of_speech)

This pattern is consistent with: prompt was tokenized to 48 tokens (not including start_of_speech), then start_of_speech was added explicitly, then speech_tokens. The masking incorrectly ignored start_of_speech.

---

## 4. Recommended Fixes

### Fix 1: Update build_prompt() in training.py

**Before:**
```python
prompt_parts.extend([
    text,
    "<|eot_id|>",
    "<|end_of_human|>",
    "<|start_of_ai|>",
    "<|start_of_speech|>",  # ⚠️ Remove this
])
```

**After:**
```python
prompt_parts.extend([
    text,
    "<|eot_id|>",
    "<|end_of_human|>",
    "<|start_of_ai|>",
    # No start_of_speech here; it's generated, not in prompt
])
```

**Rationale:** start_of_speech is the first token the model generates in the audio span, not part of the context/prompt.

---

### Fix 2: Update label construction in __getitem__() in training.py

**Before:**
```python
labels = (
    [-100] * (text_length + 1) +  # Text + start_of_speech
    speech_tokens.tolist() +
    [BODHAN_END_OF_SPEECH]
)
```

**After:**
```python
labels = (
    [-100] * text_length +         # Only text tokens masked
    [BODHAN_START_OF_SPEECH] +     # start_of_speech IS supervised
    speech_tokens.tolist() +
    [BODHAN_END_OF_SPEECH]
)
```

**Rationale:** The model must learn to generate start_of_speech token after seeing the prompt ending with `<|start_of_ai|>`. This requires supervising it in the labels.

---

## 5. Verification Plan After Fixes

1. Re-run dry_run_training.py
   - Expect: text_length = 48 (unchanged)
   - Expect: labeled start_of_speech with token ID, not -100
   - Expect: supervised positions = 471 (469 speech + 1 start + 1 end) OR 470 if end is not supervised

2. Check batch structure
   - Verify input_ids shape is correct
   - Verify attention_mask is correct
   - Verify labels have correct mix of -100 and token IDs

3. Spot-check label alignment
   - Print first 50 labels
   - Verify text portion is -100
   - Verify position 49 (start_of_speech index) has value 128257

---

## 6. Unknowns (Still Cannot Verify Without Source Code)

| Item | Status | Impact |
|------|--------|--------|
| Exact SNAC token interleaving order within frame | ❌ Unknown | Medium (affects audio quality, but serialization already validated on 10 samples) |
| Official Bodhan fine-tuning procedure | ❌ Unknown | Low (we use standard causal LM; reasonable assumption) |
| Exact loss weighting for start/end speech tokens | ❌ Unknown | Low (standard causal LM loss applies equally) |
| Whether end_of_speech should be supervised or not | ❌ Unknown | Low (conservative: supervise it; model learns exact sequence) |

---

## 7. Confidence Assessment

**Overall Confidence:** MEDIUM-HIGH

### High Confidence (>90%)
- ✅ Token ID constants are correct
- ✅ SNAC codebook layout is correct
- ✅ Speaker/style text encoding is correct
- ✅ Causal LM structure is sound
- ✅ Prompt should end at `<|start_of_ai|>`, not include `<|start_of_speech|>`

### Medium Confidence (70-90%)
- 🟡 Label masking: Should supervise start_of_speech (nearly certain, but not explicitly shown in Bodhan docs)
- 🟡 Padding strategy: Reasonable, but no explicit Bodhan training docs to verify

### Low Confidence (<70%)
- 🔴 Exact fine-tuning hyperparameters (learning rate, epochs, batch size)
- 🔴 Optimal sequence length distribution
- 🔴 Whether end_of_speech needs special handling

---

## 8. Blockers Before Training

### Must Fix (Blocking)
- ❌ Fix prompt construction (remove start_of_speech)
- ❌ Fix label masking (supervise start_of_speech)

### Should Verify (Non-blocking)
- ⚠️ Run dry-run after fixes
- ⚠️ Spot-check generated batch labels

### Does NOT Block (Already Satisfied)
- ✅ No model weights downloaded
- ✅ No training started
- ✅ SNAC serialization validated

---

## 9. Files to Modify

1. **src/marathi_tts/training.py**
   - Line ~60: Remove `"<|start_of_speech|>"` from prompt_parts
   - Line ~249: Change label construction to supervise start_of_speech

2. **docs/training-design.md** (after fixes are confirmed)
   - Add section: "Verified vs Assumed vs Unknown"
   - Clarify prompt structure: prompt ends at `<|start_of_ai|>`
   - Clarify labels: start_of_speech IS supervised
   - Update assumptions section with findings

---

**Audit Complete.** Ready to apply fixes and re-validate.
