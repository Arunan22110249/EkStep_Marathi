# EkStep Marathi TTS — Copilot Instructions

## Project

This repository implements the AI Research Engineer take-home assignment:
fine-tune the Bodhan AI TTS model on Marathi.

The target model is:
`bodhan-ai/indic-speak`

The assignment evaluates:

1. An end-to-end working fine-tuning run.
2. Code structure and quality.
3. Engineering/research judgment.
4. Problem-solving and documentation.

Model performance/metrics are explicitly not the primary evaluation criterion.

## Current environment

* OS: Windows 11
* Python: 3.10.21
* Conda environment: `ekstep_marathi_tts`
* GPU: Intel Iris Xe Graphics
* No NVIDIA CUDA GPU is available.
* Therefore do not assume CUDA availability.
* Keep local development CPU-compatible where practical.
* Do not install large or unnecessary dependencies without justification.

## Critical model constraint

Do NOT replace Bodhan Indic-Speak with another TTS model.

Do NOT invent an "official" Bodhan fine-tuning procedure.

The public Bodhan Indic-Speak release currently provides model/inference material but does not provide a public official fine-tuning recipe. If implementing fine-tuning based on the public model architecture, clearly distinguish reconstructed/our implementation from official Bodhan training code.

## Engineering principles

* Prefer simple, reproducible implementations.
* Make the smallest change necessary for each task.
* Do not rewrite unrelated files.
* Do not silently change architecture or dependencies.
* Do not download model weights or datasets unless explicitly requested.
* Do not commit model weights, datasets, generated audio, checkpoints, caches, or secrets.
* Keep large artifacts outside Git.
* Use configuration files instead of hard-coded training parameters.
* Add clear error messages and validation.
* Preserve reproducibility through explicit versions/configuration.
* Prefer deterministic behavior where practical.

## Before modifying code

1. Inspect the relevant existing files.
2. State briefly what you intend to change.
3. Make only the requested change.
4. Validate the change with the smallest appropriate command.

## Terminal usage

Do not run destructive commands.
Do not delete files or directories unless explicitly requested.
Do not install packages globally.
Use the active `ekstep_marathi_tts` Conda environment.

## Research integrity

Never fabricate:

* model capabilities
* training procedures
* dataset properties
* experimental results
* benchmark results
* citations
* official recommendations

When information is unavailable, explicitly say so.

## Project structure

Use the existing structure:

configs/
data/
docs/
experiments/
outputs/
scripts/
src/marathi_tts/

Keep application/training logic under `src/marathi_tts/`.
Keep executable entry points under `scripts/`.
Keep experiment configurations under `configs/`.
Keep documentation under `docs/`.

## Documentation

Every significant engineering decision should be documented with:

* what was done
* why it was done
* alternatives considered where relevant
* limitations
* reproducibility instructions

Do not claim that training succeeded until an actual training run has been executed and verified.

## Important workflow rule

Work incrementally.

Do NOT attempt to implement the entire project in one request.

For each task:

* inspect
* plan
* implement
* validate
* summarize changes
* stop

Wait for the next task.
