# Project plan

## Assignment objective

Implement the AI Research Engineer take-home assignment for Marathi TTS by fine-tuning the Bodhan AI TTS model on Marathi, while keeping the work reproducible, conservative, and grounded in the available project structure and constraints.

## Target model

- bodhan-ai/indic-speak

## Known model facts that can be verified from current project/source information

- The repository explicitly states that the assignment is to fine-tune Bodhan AI TTS on Marathi.
- The target model is bodhan-ai/indic-speak.
- The project instructions state that this model must not be replaced with another TTS model.
- The instructions explicitly warn against inventing an "official" Bodhan fine-tuning procedure.
- The public Bodhan Indic-Speak release currently provides model/inference material but does not provide a public official fine-tuning recipe.
- If fine-tuning is implemented from the public model architecture, the result must be clearly identified as a reconstructed implementation rather than official Bodhan training code.
- The repository is only a minimal scaffold at this stage: directories exist for configs, data, docs, experiments, outputs, scripts, and src/marathi_tts, but there is no implementation code yet.
- The environment is Windows 11 with Python 3.10.21 and a Conda environment named ekstep_marathi_tts.
- The available GPU is Intel Iris Xe Graphics; there is no NVIDIA CUDA GPU available.
- The project instructions require keeping local development CPU-compatible where practical and avoiding assumptions of CUDA availability.

## What still needs to be verified before implementation

- The exact public model repository or artifact layout for bodhan-ai/indic-speak and whether local code or checkpoints are available in this workspace.
- The model architecture, tokenizer, audio format, and training/inference entry points used by the public release.
- The exact dataset source, language coverage, license, and data format that will be used for Marathi fine-tuning.
- Whether a suitable Marathi dataset already exists in the data/ directory or must be sourced externally.
- The dependency stack required for the public model implementation and for fine-tuning in a CPU-first environment.
- Whether the execution environment can support any non-CPU recipes or whether all work must remain CPU-compatible.
- Whether any existing scripts or experiment configs in the repository define evaluation or training conventions that should be followed.

## Proposed development phases

1. Model/inference inspection
   - Inspect the public model release, architecture, and inference code paths for bodhan-ai/indic-speak.
   - Identify what is required to run inference locally and what assumptions are embedded in the public implementation.

2. Dependency selection
   - Choose the minimal dependency set needed to load the model, process audio/text, and run fine-tuning or evaluation.
   - Prefer simple, reproducible dependencies and avoid unnecessary additions.

3. Dataset selection
   - Identify a Marathi TTS dataset or subset that is compatible with the model and task.
   - Confirm format, licensing, and availability before using any data.
   - **Selected Dataset:** SPRINGLab/IndicTTS_Marathi (10,939 samples, 48 kHz, 4 native speakers)
   - **Rationale:** TTS-native dataset from IIT Madras; higher sample count and quality than alternatives; requires 48→24 kHz resampling only
   - **Details:** See docs/dataset-inspection.md for full metadata, schema, preprocessing pipeline, and risk analysis

4. Preprocessing
   - Convert text and audio into a consistent format for training.
   - Validate audio quality, duration, transcription alignment, and label correctness.

5. Fine-tuning implementation
   - Implement the minimal training pipeline for the reconstructed model setup.
   - Clearly separate any reconstructed training logic from official Bodhan training code or claims.

6. Smoke test
   - Run a minimal end-to-end check to confirm that the pipeline loads, preprocesses data, and can execute a short training step or dry run.

7. Training
   - Run the actual fine-tuning job with explicit configuration and reproducible settings.
   - Keep logs and saved configuration in a structured way.

8. Inference/evaluation
   - Test generated Marathi speech output after fine-tuning.
   - Evaluate basic functionality and sanity-check results without overstating performance claims.

9. Documentation
   - Record what was done, why, the constraints considered, and reproduction steps.
   - Include limitations, unresolved uncertainties, and the exact configuration used for training and inference.

## Hardware constraints

- No NVIDIA CUDA GPU is available in the current environment.
- The machine has an Intel Iris Xe Graphics GPU, which is not assumed to be CUDA-capable.
- The project instructions explicitly require CPU-friendly development choices where practical.
- Any implementation should therefore avoid assuming GPU-accelerated training or CUDA-only tooling unless explicitly verified.

## Reproducibility requirements

- Preserve the existing repository structure and avoid unnecessary rewrites.
- Keep application and training logic under src/marathi_tts/ and executable entry points under scripts/.
- Use configuration files instead of hard-coded training parameters.
- Do not download model weights or datasets unless explicitly requested.
- Do not commit model weights, datasets, generated audio, checkpoints, caches, or secrets.
- Keep large artifacts outside Git.
- Document significant engineering decisions, including rationale, alternatives considered, limitations, and reproducibility instructions.
- Do not claim training succeeded without an actual training run and verification.
