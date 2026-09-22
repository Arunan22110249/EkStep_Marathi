# Bodhan Indic-Speak Speech Token Contract

## 1. Prompt structure

OFFICIAL BODHAN DOCUMENTATION:
- The public model card shows the inference API as `tts(text, speaker="Amit")` and `tts.save("output.wav", wav)`.
- The model card also provides a CLI example: `python inference.py --text "..." --speaker Amit --style ANGER`.
- The model card explicitly states that `speaker` is a voice selector and that unseen speaker names lead to worse voice quality.
- Source: Hugging Face model card section “How to Use”; visible in the public model card.

PUBLIC IMPLEMENTATION EVIDENCE:
- The public repo tree lists `inference.py`, `tokenizer.json`, `tokenizer_config.json`, `token_contract.md`, `voices.md`, and a `vocos/` directory.
- Source: Hugging Face repo tree preview for `bodhan-ai/indic-speak`.

INFERENCE/UNKNOWN:
- The exact prompt template is not publicly visible, because the gated `inference.py` and `token_contract.md` contents could not be accessed in this environment (HTTP 401 from the public URLs).
- The exact ordering of text, speaker, style, and any speech-control tokens is therefore unknown.
- No public special text/prompt token names were visible in the accessible documentation.

## 2. Text and control tokens

OFFICIAL BODHAN DOCUMENTATION:
- The public model card shows plain text input strings in Marathi/Hindi, e.g. `tts("प्रकाश की चाल लगभग तीन लाख किलोमीटर प्रति सेकंड होती है।", speaker="Amit")`.
- The generated arguments include `style`, `temperature`, `top_p`, `top_k`, `max_new_tokens`, `seed`, and `stock`.
- The public model card says `style` is preview-quality and may respond to emotion labels or descriptive phrases.
- The model card explicitly says a `speaker` name must have been seen in training for best voice quality.
- Source: Hugging Face model card section “How to Use”; “Voices” note.

PUBLIC IMPLEMENTATION EVIDENCE:
- `tokenizer.json` and `tokenizer_config.json` are listed in the public model repo, which confirms that the model release includes tokenizer files.
- `token_contract.md` is also listed, which confirms a public token contract file exists in the released implementation.
- Source: Hugging Face repo tree preview for `bodhan-ai/indic-speak`.

INFERENCE/UNKNOWN:
- No exact token IDs, token ranges, or special token names were visible from the accessible public materials.
- No exact special token IDs (BOS/EOS/PAD/title/style/speaker/speech-start/speech-end) could be verified.
- The actual token contract contents were not accessible because the file was gated.

## 3. Speech control tokens

OFFICIAL BODHAN DOCUMENTATION:
- `speaker` is documented in the public API and is described as a required voice selector.
- `style` is documented as an optional control field for emotion or descriptive phrasing.
- The public model card states that style control is preview-quality and not yet reliable.
- Source: Hugging Face model card, “How to Use” and “Limitations”.

PUBLIC IMPLEMENTATION EVIDENCE:
- `voices.md` is listed in the public model tree, which implies a voice registry or voice metadata file is part of the public release.
- Source: Hugging Face repo tree preview for `bodhan-ai/indic-speak`.

INFERENCE/UNKNOWN:
- The exact speaker token format is unknown.
- The exact style token format is unknown.
- There is no publicly visible evidence of a speech start token or speech end token in the accessible documentation.
- No exact IDs or token names for speech-control behavior were verified.

## 4. SNAC token representation

OFFICIAL BODHAN DOCUMENTATION:
- The model card explicitly states that the system uses the SNAC quantizer and a finetuned Vocos decoder.
- The public model card also states that the final output is a float32 NumPy array at 24 kHz.
- Source: Hugging Face model card section “License / Terms of Use”; “How to Use”.

PUBLIC IMPLEMENTATION EVIDENCE:
- The public repo tree contains a `vocos/` directory and `model.safetensors`.
- Source: Hugging Face repo tree preview for `bodhan-ai/indic-speak`.

INFERENCE/UNKNOWN:
- The exact SNAC token IDs are not publicly visible.
- The exact SNAC token naming scheme is not publicly visible.
- No public SNAC vocabulary table, codebook table, or mapping file was available in the accessible evidence.
- The exact relationship between LM output IDs and SNAC code IDs remains unverified.

## 5. Codebook structure

OFFICIAL BODHAN DOCUMENTATION:
- There is no public Bodhan statement in the accessible sources that gives the exact number of SNAC codebooks.
- There is no public Bodhan statement in the accessible sources that gives codes-per-codebook values.

PUBLIC IMPLEMENTATION EVIDENCE:
- The public repo tree includes a `vocos/` directory and model assets, but no public codebook definition or SNAC config file was visible in the accessible preview.
- The visible tree does not include a SNAC config or codec metadata file.

INFERENCE/UNKNOWN:
- Number of SNAC codebooks: UNKNOWN.
- Codes per codebook: UNKNOWN.
- Interleaving order of codebooks: UNKNOWN.
- Whether LM outputs are flattened or structured multi-codebook tokens: UNKNOWN.
- The exact output indexing scheme of the codec tokens is unknown.

## 6. Token-to-SNAC conversion

OFFICIAL BODHAN DOCUMENTATION:
- No exact token-to-SNAC conversion logic is documented in the accessible public Bodhan model card.

PUBLIC IMPLEMENTATION EVIDENCE:
- The public repo tree includes `inference.py`, which is the implementation file that obviously handles generation and output conversion, but its contents were not accessible because the public file was gated.
- Source: Hugging Face repo tree preview for `bodhan-ai/indic-speak`.

COMMUNITY IMPLEMENTATION REFERENCE:
- No accessible community mirror or conversion of Bodhan `token_contract.md` was found in the current public tools/search results.
- This environment did not yield a usable non-official reference for the exact token conversion logic.

INFERENCE/UNKNOWN:
- The exact mapping from LM token IDs to SNAC codes is unknown.
- The exact conversion step from generated IDs to codec tokens is unknown.
- Whether any remapping, grouping, or residual ordering is used is unknown.

## 7. SNAC-to-Vocos representation

OFFICIAL BODHAN DOCUMENTATION:
- The public model card explicitly identifies SNAC and Vocos as components of the generation stack.
- The model card says the final output is a float32 NumPy waveform at 24 kHz.

PUBLIC IMPLEMENTATION EVIDENCE:
- The repo tree includes a `vocos/` directory, which indicates the decoder package or decoder artifacts are part of the release.
- Source: Hugging Face repo tree preview.

INFERENCE/UNKNOWN:
- The exact latent representation passed to Vocos is unknown.
- The exact SNAC-to-Vocos projection or decoding stage is unknown.
- Whether the decoder input is residual codebook tensors, a flattened latent sequence, or another representation is unknown.

## 8. Audio representation

OFFICIAL BODHAN DOCUMENTATION:
- The public model card provides a direct output example showing `tts.save("output.wav", wav)` and a comment `# float32 numpy @ 24 kHz`.
- Source: Hugging Face model card, “How to Use”.

PUBLIC IMPLEMENTATION EVIDENCE:
- `inference.py` is listed as a public implementation artifact; it is consistent with waveform generation and save logic.
- Source: Hugging Face repo tree preview.

INFERENCE/UNKNOWN:
- Exact channel count is unknown.
- Exact waveform normalization is unknown.
- Any internal resampling or time-scaling behavior is unknown.

## 9. What is directly verified

OFFICIAL BODHAN DOCUMENTATION:
- Model: `bodhan-ai/indic-speak`.
- Category: multilingual TTS model.
- Supported languages include Marathi in the production set.
- Base model: `meta-llama/Llama-3.2-3B`.
- System components include SNAC and a finetuned Vocos decoder.
- Inference API accepts text plus a `speaker` argument and optional `style` argument.
- Final output is saved as `.wav` and described as a float32 NumPy array at 24 kHz.
- Source: Hugging Face model card.

PUBLIC IMPLEMENTATION EVIDENCE:
- Public repo tree includes `inference.py`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `token_contract.md`, `voices.md`, and `vocos/`.
- Source: Hugging Face repo tree preview.

COMMUNITY IMPLEMENTATION REFERENCE:
- None found in the current accessible public tools/search results that was both accessible and usable as a direct implementation reference.

## 10. What remains unknown for training

INFERENCE/UNKNOWN:
- Vocabulary ranges and exact tokenizer IDs.
- Special/control token IDs and ranges.
- SNAC token range and token naming scheme.
- Number of SNAC codebooks.
- Codes per codebook.
- Conditioning/paralinguistic token range.
- Speech start/end token IDs.
- Prompt structure beyond the high-level `text + speaker + style` public API.
- Relationship between LM output token IDs and SNAC code IDs.
- SNAC quantizer interface and exact Vocos input representation.
- Final audio sample rate is verified at 24 kHz, but channel layout, normalization, and internal resampling remain unknown.
- No public training objective, fine-tuning recipe, optimizer configuration, dataset preprocessing pipeline, or LoRA config was verified.

This document intentionally does not assert any training objective, fine-tuning recipe, or optimizer setting beyond the public model card and repo preview. The accessible public evidence supports inference and model architecture only; it does not provide a verified official training implementation.
- Any public dataset or preprocessing code for Marathi or other languages.

These items are unknown from the public implementation accessible here, and they must not be treated as official Bodhan fine-tuning details unless the official Bodhan source explicitly states them.

Source references:
- Model card and repo tree preview reviewed here.
- The actual public implementation files `inference.py` and `token_contract.md` were not accessible in this environment due to HTTP 401 gating.
