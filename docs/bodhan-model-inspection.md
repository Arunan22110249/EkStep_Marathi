# Bodhan Indic-Speak Model Inspection

## 1. Model identity

- Model name: bodhan-ai/indic-speak.
- Public model card description: a text-to-speech model for 22 Indian languages and English, positioned for educational content and explanatory reading; the model page describes it as a general-purpose voice engine for classroom-first use cases.
- Public model card identifies the base model as meta-llama/Llama-3.2-3B and says the model is a finetuned version of that base model.
- Public metadata also lists the model as a text-to-speech model with Safetensors and a model size of roughly 3B parameters.
- Public file tree preview for the model repository includes `config.json`, `generation_config.json`, `inference.py`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `token_contract.md`, `voices.md`, and a `vocos` directory.
- Public model card states the model supports production languages including Marathi, and the language list includes Marathi in the production set.

Sources:
- Hugging Face model card: bodhan-ai/indic-speak, “What is Indic-Speak?”, “Languages Supported”, “Model tree for bodhan-ai/indic-speak”, and license/terms sections.
- Hugging Face repository tree preview: bodhan-ai/indic-speak/tree/main, visible file list and attributes.

## 2. Architecture

Officially documented information:

- The model is explicitly identified as a text-to-speech model for multilingual Indian-language speech generation.
- The model card states: “The base model is Llama-3.2-3B, and this system also uses the SNAC Quantizer and a finetuned Vocos decoder.”
- The model page also states the model uses a transformer-based architecture as a fine-tuned Llama model, with an audio stack built around SNAC and Vocos.
- The file tree preview includes a `vocos` directory and a `model.safetensors` checkpoint, indicating that the public release includes a decoder component beyond the base LLM weights.
- The public model page says the model supports multiple voices and style controls, with the presence of the `voices.md` file indicating speaker-level conditioning or metadata.

Inferred from implementation and observed artifacts:

- The public release strongly suggests an end-to-end TTS stack where a language model generates symbolic or discrete audio-token sequences, which are then reconstructed into waveform audio via a neural codec/decoder pipeline.
- The presence of `SNAC` and `Vocos` in the public description strongly indicates a neural codec (discrete acoustic tokenization) plus a waveform decoder, rather than a direct waveform-output language model.
- The file `inference.py` and the model card’s “How to Use” examples indicate there is a callable `TTS` object that shares a common inference interface across text input and speaker/style arguments.
- The model is not a simple single-stage text-to-waveform end-to-end model in the public description; it is described as a system built from model weights plus audio-generation helper components.

Information that is missing:

- The exact architecture of the model head and the precise text-to-codec generation mechanism are not publicly described in the visible files.
- The exact placement of speaker and style conditioning inside the architecture is not visible from the public metadata.
- The exact configuration of SNAC and Vocos is not available in the public file list, despite their names appearing in the documentation.

Sources:
- Hugging Face model card: “License / Terms of Use” section (explicit base model and SNAC/Vocos provenance).
- Hugging Face repository tree preview: `inference.py`, `model.safetensors`, `vocos` directory, `config.json`.

## 3. Text/token representation

Officially documented information:

- The public file tree includes `tokenizer.json`, `tokenizer_config.json`, and `token_contract.md`, which indicates the model exposes a public tokenizer and a documented token contract.
- The model card includes usage examples such as `tts("प्रकाश की चाल लगभग तीन लाख किलोमीटर प्रति सेकंड होती है।", speaker="Amit")`, showing that the model accepts plain text plus explicit speaker metadata as inference input.
- The usage examples also show generation arguments such as `style`, `temperature`, `top_p`, `top_k`, `max_new_tokens`, `seed`, and `stock`.
- The public model card explicitly states: “`speaker` must be a name the model saw in training — an unseen name does not error, it just gives an averaged, worse voice.”
- The `voices.md` file is present in the model tree; this indicates speaker identities or voice metadata are part of the supported public interface.

Inferred from implementation and observed artifacts:

- The model is likely conditioned on plain text tokens plus speaker metadata and optional style parameters, rather than a purely raw-audio-generation interface.
- The presence of `token_contract.md` strongly suggests the public release documents a contract that defines how text tokens or prompts are structured for generation.
- Because the tokenizer files are present, the model likely uses a standard decoder-based tokenization scheme consistent with the underlying Llama base model or a custom patched implementation.

Information that is missing:

- The exact token format and schema are not accessible in the public information reviewed here, because the actual `token_contract.md` content is not visible without access to the gated model files.
- The exact text normalization, language tag handling, or code-mix handling rules are not visible from the public metadata reviewed here.
- The exact tokenizer class and vocabulary details are not available from the public preview.

Sources:
- Hugging Face model card: “How to Use” examples and the “speaker” guidance.
- Hugging Face repository tree preview: `tokenizer.json`, `tokenizer_config.json`, `token_contract.md`, `voices.md`.

## 4. Speech/audio representation

Officially documented information:

- The public model card’s inference example states: `tts.save("output.wav", wav)` and that `wav` is a float32 NumPy array at 24 kHz.
- The model card further notes the output is a waveform saved as a `.wav` file, and the public example shows a simple save call.
- The license/terms section documents that the model uses the SNAC quantizer and a finetuned Vocos decoder.
- The model is described as a speech synthesizer that can generate natural-sounding output for multilingual text.

Inferred from implementation and observed artifacts:

- The output is not a plain text transcription; it is generated as speech waveform data.
- The combination of a quantizer and decoder suggests the model generates a discrete audio representation first, then reconstructs the waveform via Vocos.
- The model likely operates with a 24 kHz waveform representation, as described in the usage example.

Information that is missing:

- The specific codec configuration, latent dimensions, frame rate, and waveform reconstruction details are not publicly visible from the accessible file list or model card excerpt.
- The exact public audio token format is not described in the visible metadata.

Sources:
- Hugging Face model card: “How to Use” code snippet showing waveform output and `save("output.wav", wav)`.
- Hugging Face model card: license/terms section naming SNAC and Vocos.

## 5. Inference pipeline

Officially documented information:

- The public model card provides a direct Python example:
  - `from inference import TTS`
  - `tts = TTS("bodhan-ai/indic-speak-preview-v2")`
  - `wav = tts(text, speaker="Amit")`
  - `tts.save("output.wav", wav)`
- The model card also includes a CLI example: `python inference.py --text "..." --speaker Amit --style ANGER`.
- The model card documents generation parameters including `temperature`, `top_p`, `top_k`, `max_new_tokens`, `seed`, and `stock`.
- The model card states that `speaker` names matter for voice identity: “a speaker name the model saw in training” yields better voice fidelity than unseen names.
- The model card explicitly says style control is preview-quality and not fully reliable, which is relevant to the expected inference behavior.

Inferred from implementation and observed artifacts:

- The inference interface is centered around a `TTS` object that accepts text and speaker metadata and emits waveform output.
- The model’s inference path is likely: prompt text + voice metadata -> LLM generation -> audio-token or codec-token generation -> SNAC/Vocos decoding -> waveform output.
- The CLI and Python examples suggest that inference is intentionally designed to be simple and easy to run from a script or command line.

Information that is missing:

- The exact code path inside `inference.py` is not visible in the public review here, so the exact sequence of token generation, style conditioning, and decoding cannot be conclusively reconstructed without access to the gated files.
- The exact prompt format and any hidden control tokens are not available from the accessible metadata.

Sources:
- Hugging Face model card: “How to Use” section and the `style`/`speaker` notes.
- Hugging Face file tree preview: `inference.py`, `generation_config.json`, `tokenizer.json`, `voices.md`.

## 6. Repository structure

From the public Hugging Face tree preview, the model repository appears to include the following core artifacts:

- `README.md`
- `config.json`
- `generation_config.json`
- `inference.py`
- `model.safetensors`
- `token_contract.md`
- `tokenizer.json`
- `tokenizer_config.json`
- `voices.md`
- `vocos/` directory
- `banner.png`
- `Bodhan_AI_Open_Model_License.md` / `indic-open-license.md` (license files)
- `.gitattributes`

Additional public metadata also indicates:

- The checkpoint is a finetuned model with a base model of `meta-llama/Llama-3.2-3B`.
- The file tree preview lists a `vocos` folder with a commit reference that mentions `v3 step140k -> v9_gen step200k (EMA)`, suggesting the decoder checkpoint or model artifact is present in the repo.
- The model card advertises both finetune and quantization entries in the model tree, but those are not directly inspectable from the visible public tree snapshot.

Important limitation:

- The public GitHub repository URL for `github.com/bodhan-ai/indic-speak` returned HTTP 404 in the inspection attempt, so the public inspection here is limited to the Hugging Face model card and file-tree preview rather than a full public source repository.

Sources:
- Hugging Face repository tree preview for bodhan-ai/indic-speak.
- Hugging Face model card: “Model tree for bodhan-ai/indic-speak”.

## 7. Available checkpoints/artifacts

Officially visible checkpoint/artifact information:

- `model.safetensors` is listed as a public model artifact for the model and is described as a 6.6 GB Safetensors file.
- `config.json` and `generation_config.json` are present in the public tree, indicating model configuration and generation settings are packaged with the checkpoint.
- `tokenizer.json` and `tokenizer_config.json` are present, indicating the tokenizer artifacts are part of the public release.
- `vocos` artifacts are present in the public file tree, indicating a decoder/checkpoint component is bundled in the release.
- The public model card also indicates there are quantized versions and finetuned variants in the broader model family, but those are not directly accessible in the visible file tree snapshot.

Important limitations:

- The model card and tree preview do not show a public training checkpoint or a public open fine-tuning script.
- The public listing does not provide a dataset manifest or dataset preparation file in the visible tree.
- No training configuration file is visible in the public tree preview.

Sources:
- Hugging Face repository tree preview: `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocos/`.
- Hugging Face model card: “Model tree for bodhan-ai/indic-speak” and file size metadata.

## 8. Training/fine-tuning support

Officially documented information:

- The public license text explicitly permits modification and fine-tuning: “You are free to ... change it — fine-tune, distill, quantize, merge, or otherwise build on it.”
- The public model card states the base model and notes the presence of a finetuned model, confirming that fine-tuning is part of the model’s intended use.
- The model card’s limitations section does not describe a public official training recipe or training pipeline.

What the public repository appears to contain:

- Inference code: `inference.py` is present.
- Model weights and tokenizer/config files: present.
- Speaker metadata: `voices.md` is present.
- Audio decoder component: `vocos` is present.

What the public repository does not appear to contain from the accessible preview:

- training code
- fine-tuning code
- dataset preprocessing code
- training configuration
- LoRA/PEFT configuration
- training commands
- trainer scripts
- dataset manifests

This is a significant limitation: the public release is an inference model package, not a public training repository.

Sources:
- Hugging Face model card: “License / Terms of Use” section and use-case guidance.
- Hugging Face repository tree preview: visible file list and the absence of standard training artifacts such as `train.py`, `training_args`, `dataset`, or LoRA config files.

## 9. What is officially documented

The following items are clearly documented in the public materials reviewed here:

- The target model is bodhan-ai/indic-speak.
- It is a multilingual TTS model for 22 Indian languages and English.
- It is built for classroom-first educational and explanatory speech use cases.
- The public model card describes the model as a finetuned Llama-3.2-3B model.
- The public model card identifies SNAC and a finetuned Vocos decoder as components of the system.
- The public model card provides example inference code and CLI usage.
- The public model card states that `speaker` names matter and that model output quality depends on seen speaker identities.
- The public model card states that style control is preview-quality and not fully reliable.
- The public model card documents the supported languages and voice-related features.
- The public model card provides public license terms authorizing modification and fine-tuning.

These statements are from the visible model card and tree preview; they should be treated as the boundary of the verified public information.

Sources:
- Hugging Face model card sections: “What is Indic-Speak?”, “Languages Supported”, “How to Use”, “Limitations”, “License / Terms of Use”, and “Model tree for bodhan-ai/indic-speak”.

## 10. What must be reconstructed

Because the public model release does not expose an official fine-tuning recipe, the following items must be reconstructed from inference code, architecture notes, and surrounding public metadata rather than taken as official implementation guidance:

- the exact fine-tuning objective and loss formulation
- the exact training loop and checkpointing logic
- the exact data pipeline for text and audio preprocessing
- the exact speaker conditioning schema used during training
- the exact style-conditioning design and token format
- the exact training configuration and command sequence
- the exact relationship between the base Llama model and the audio decoder stack during training
- any LoRA/PEFT, adapter, or efficient-tuning method used for a fine-tune of the released model
- the Marathi-specific data preparation and normalization strategy

This is not a claim that those details are absent because they are impossible; only that they are not publicly documented in the available model release artifacts reviewed here.

Sources:
- Public model release preview, which exposes inference artifacts and licensing but not a public training stack.
- Model card statements that warn against inventing an official Bodhan fine-tuning procedure and explicitly distinguish the public release from official training code.

## 11. Implications for Marathi fine-tuning

- The public model supports Marathi as a production language, which makes Marathi fine-tuning a meaningful task in principle.
- The model is publicly released as a TTS system with inference support, but it is not accompanied by an official public fine-tuning recipe in the accessible materials.
- That means the assignment is best treated as a reconstruction exercise grounded in public architecture and inference behavior, not as a direct official Bodhan training procedure.
- Because no public training code or training config is visible, any Marathi fine-tuning implementation must be justified as a reconstructed design based on the public model architecture and the surrounding metadata.
- The public model page does not provide dataset details or preprocessing instructions, so those must be identified separately before implementation.
- The user environment constraint remains relevant: no NVIDIA CUDA GPU is available, so any local implementation should not assume CUDA-only execution.

Sources:
- Model card section listing Marathi as a production language.
- Public model card “License / Terms of Use” and related guidance.
- Project-local instructions in the workspace stating the task is to fine-tune the model on Marathi and not to invent an official Bodhan fine-tuning procedure.

## 12. Open implementation questions

- What is the exact training objective used by the public model, beyond the high-level description of a TTS system?
- What does the public `token_contract.md` define in detail, and how is it used during generation?
- What exact speaker metadata and style tags are used in the public training pipeline?
- Is the audio representation discrete codec tokens, continuous features, or another representation?
- What is the exact training configuration, dataset pipeline, and checkpoints used by Bodhan AI’s internal training process?
- Is there a public LoRA/PEFT recipe, or is the model intended to be fine-tuned in a fully parameter-updating manner?
- What specific preprocessing pipeline is required for Marathi text and speech before fine-tuning?
- What is the exact public repository structure for a full training implementation? The current public preview suggests an inference package rather than a full training repo.
- Which model variant should be used for a Marathi-focused fine-tune when both base model and quantized variants exist?

Sources:
- Public model card and file tree preview.
- Public repository inspection results, including the missing training assets and the absence of a public training configuration.
