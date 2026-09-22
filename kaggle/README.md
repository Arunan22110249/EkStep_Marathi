# Kaggle GPU Execution for Bodhan TTS Smoke Test

This directory contains scripts and configuration for running the Bodhan TTS fine-tuning smoke test on **Kaggle's free GPU environment**.

---

## Quick Start (5 minutes)

```bash
# 1. Authenticate Kaggle (one-time)
kaggle auth login

# 2. Push kernel to Kaggle
kaggle kernels push -p .

# 3. Wait for execution (~5-10 minutes)
# 4. Download results
kaggle kernels output ekstep-marathi-bodhan-smoke-test -p ./downloads/

# 5. View results
cat downloads/kaggle_smoke_test_results.json
```

---

## Detailed Setup Instructions

### **STEP 1: Kaggle CLI Authentication**

#### 1.1 Install Kaggle CLI
```bash
pip install kaggle
```

#### 1.2 Get Your Kaggle API Token

1. Go to https://www.kaggle.com/settings/account
2. Click **"Create New API Token"**
3. This downloads a file named `kaggle.json`
4. Move it to the correct location:

**Windows:**
```bash
move kaggle.json C:\Users\<YourUsername>\.kaggle\
```

**Mac/Linux:**
```bash
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 1.3 Verify Authentication
```bash
kaggle kernels list
```

If this shows your kernels without errors, you're authenticated ✓

---

### **STEP 2: Push Kernel to Kaggle**

#### 2.1 Update Your Username in kernel-metadata.json

Open `kernel-metadata.json` and ensure the `id` field is unique to your Kaggle account:

```json
{
  "id": "your-username-bodhan-smoke-test",
  "title": "EkStep Marathi - Bodhan TTS Smoke Test (GPU)"
}
```

#### 2.2 From VS Code Terminal

Navigate to the `kaggle/` directory:

```bash
cd kaggle/
```

Push the kernel:

```bash
kaggle kernels push -p .
```

**Expected output:**
```
Kernel version <number> successfully created.
View your kernel at https://www.kaggle.com/code/your-username/your-username-bodhan-smoke-test
```

#### 2.3 Verify on Kaggle

Visit the URL shown in the output to confirm the kernel appears on Kaggle.

---

### **STEP 3: Enable GPU and Add Secrets**

#### 3.1 Enable GPU in Kernel Settings

1. Open your kernel on Kaggle (from step 2.3)
2. Click **"Settings"** (gear icon, top right)
3. Scroll to **"Accelerator"**
4. Select **"GPU"** (P100 or higher recommended)
5. Click **"Save"**

#### 3.2 Provide the Hugging Face Token

Authentication is resolved at runtime in this priority order:

1. **Kaggle Secret `HF_TOKEN`** — if the `kaggle_secrets` package is available
   (i.e. running inside a Kaggle kernel) and a secret named `HF_TOKEN` has been
   attached to the kernel, it is read via `UserSecretsClient().get_secret("HF_TOKEN")`.
   Add it via kernel **"Add-ons" → "Secrets"** in the Kaggle editor.
2. **`HF_TOKEN` environment variable** — set directly in the Kaggle runtime
   environment (e.g. via a startup script or notebook cell that exports it
   before running `run_bodhan_smoke_test.py`).
3. **`HF_TOKEN_FILE` environment variable** — points to a runtime-mounted file
   containing the token (e.g. a file added as a Kaggle Dataset/Input, or
   written by a setup step). The runner reads the token from that file.
4. **`--hf-token-file PATH` CLI argument** — an explicit path to a runtime
   token file, equivalent to `HF_TOKEN_FILE` but supplied on the command line.
   This argument must always be a **file path**, never the token value itself.
5. **Local development fallback: `secrets/hf_token.txt`** — used only when
   running the script locally (outside Kaggle) and the file exists.

**`secrets/hf_token.txt` is intentionally listed in `.gitignore` and is never
committed to GitHub or uploaded to Kaggle.** On Kaggle, this file will not be
present — use a Kaggle Secret, `HF_TOKEN`, or `HF_TOKEN_FILE`/`--hf-token-file` instead.

**How to get a token:**
- Go to https://huggingface.co/settings/tokens
- Click **"New token"**
- Name: "Bodhan Training"
- Permissions: Read (to download model)
- Copy the token

**Local development setup:**
```
secrets/hf_token.txt   # contains only the token, never committed
```

**Kaggle runtime setup (choose one):**
```
# Option A: Kaggle Secret (recommended on Kaggle)
# Kernel editor → Add-ons → Secrets → add HF_TOKEN

# Option B: environment variable
export HF_TOKEN=<token>

# Option C: runtime token file mounted into the kernel
export HF_TOKEN_FILE=/path/to/mounted/token/file

# Option D: CLI argument (file path only)
python run_bodhan_smoke_test.py --hf-token-file /path/to/mounted/token/file
```

If `kaggle_secrets` is unavailable (e.g. local execution), that step is
silently skipped and resolution falls through to the next source. If none of
these sources resolve a token, the runner fails clearly with:
```
Hugging Face authentication token not provided. Set HF_TOKEN or provide a runtime token file through HF_TOKEN_FILE.
```


The runner never prints the token value, its length, or any derived characters —
only `HF_TOKEN available: True` or `HF_TOKEN available: False`.

#### 3.3 Validate Authentication Resolution (No Token Printed)

To check which authentication source will be used without running the full
smoke test:
```
python run_bodhan_smoke_test.py --validate-auth
```
This prints only `HF_TOKEN available: True` or `False`.

---


### **STEP 4: Run the Smoke Test**

#### 4.1 Manual Execution on Kaggle

1. Open your kernel on Kaggle
2. Click the **"Run All"** button
3. Wait for execution (5–15 minutes depending on GPU availability)

The script will:
- ✓ Detect CUDA and GPU
- ✓ Clone the repository
- ✓ Install dependencies
- ✓ Authenticate with Hugging Face
- ✓ Run the smoke test (3 optimizer steps)
- ✓ Save results to JSON

#### 4.2 Monitor Progress

Check the kernel output as it runs:
- **Green checkmarks (✓)** = success
- **Red X marks (✗)** = failure
- Watch for GPU memory usage and loss values

#### 4.3 Common Issues and Solutions

**Issue: "403 Forbidden" when downloading model**
- **Cause:** HF_TOKEN not resolved or invalid
- **Solution:** Set `HF_TOKEN`, or provide `HF_TOKEN_FILE` / `--hf-token-file` pointing to a runtime token file (see Step 3.2)

**Issue: "CUDA out of memory"**
- **Cause:** GPU doesn't have enough memory
- **Solution:** This is expected feedback; note the peak memory usage for the report

**Issue: "git: not found"**
- **Cause:** Git not installed on this kernel image
- **Workaround:** The kernel includes git; if error persists, reinstall:
  ```bash
  apt-get update && apt-get install -y git
  ```

**Issue: Kernel times out (>30 minutes)**
- **Cause:** Download is very slow or model is large
- **Solution:** Check GPU availability; re-run later if GPU queue is congested

---

### **STEP 5: Retrieve and Download Results**

#### 5.1 Download Results from Kaggle

```bash
# Download all outputs
kaggle kernels output your-username/your-username-bodhan-smoke-test -p ./results/
```

This creates a `results/` directory containing:
- `kaggle_smoke_test_results.json` — All metrics and logs

#### 5.2 View Results

```bash
cat results/kaggle_smoke_test_results.json
```

Or, pretty-print with Python:

```bash
python -c "import json; f=open('results/kaggle_smoke_test_results.json'); print(json.dumps(json.load(f), indent=2))"
```

#### 5.3 Expected Output Structure

```json
{
  "execution_start": "2026-09-22T10:30:00.000000",
  "environment": {
    "python_version": "3.10.13",
    "pytorch_version": "2.1.0+cu118",
    "cuda_available": true,
    "device_name": "NVIDIA A100-PCIE-40GB",
    "supports_bf16": true,
    "recommended_dtype": "bfloat16"
  },
  "cuda_detection": {
    "cuda_available": true,
    "gpu_name": "NVIDIA A100-PCIE-40GB",
    "gpu_vram_gb": 40.0
  },
  "smoke_test_results": {
    "status": "completed",
    "success": true,
    "configuration": {
      "device": "cuda",
      "dtype": "bfloat16",
      "batch_size": 1,
      "num_steps": 3
    },
    "output": "... full training output ..."
  },
  "execution_end": "2026-09-22T10:45:00.000000"
}
```

---

## File Structure

```
kaggle/
├── run_bodhan_smoke_test.py     ← Main execution script
├── kernel-metadata.json          ← Kaggle configuration
└── README.md                      ← This file
```

---

## What the Script Does

### run_bodhan_smoke_test.py

1. **STEP 1: CUDA Detection**
   - Runs `nvidia-smi`
   - Detects GPU model and VRAM
   - Aborts if CUDA not available

2. **STEP 2: Python Environment**
   - Verifies PyTorch version
   - Checks bf16/fp16 support
   - Recommends optimal dtype

3. **STEP 3: Repository Management**
   - Clones the EkStep Marathi repository from GitHub
   - Or pulls latest changes if already exists

4. **STEP 4: Dependencies**
   - Installs packages from `requirements.txt`
   - Ensures CUDA PyTorch is installed

5. **STEP 5: Hugging Face Authentication**
   - Resolves the token from Kaggle Secret `HF_TOKEN`, `HF_TOKEN` env var, `HF_TOKEN_FILE` / `--hf-token-file`, or (locally only) `secrets/hf_token.txt`
   - Authenticates with Hugging Face using the resolved token
   - Required to download Bodhan model weights
   - Never prints the token value

6. **STEP 6: Smoke Test Execution**
   - Runs `scripts/smoke_test_training.py` with GPU configuration
   - Configuration: batch_size=1, 3 steps, bf16/float16, no quantization/LoRA
   - Records all metrics: GPU memory, loss progression, training success
   - Saves results to `outputs/kaggle_smoke_test_results.json`

---

## Configuration Parameters

Edit `run_bodhan_smoke_test.py` to change smoke test parameters:

```python
SMOKE_TEST_CONFIG = {
    "device": "cuda",               # GPU device
    "dtype": "bfloat16",            # or float16
    "batch_size": 1,                # Batch size (1 is default)
    "max_samples": 10,              # Use first 10 samples
    "num_steps": 3,                 # Optimizer steps (increase to 5 if passes)
    "learning_rate": 1e-5,          # Learning rate
    "gradient_accumulation_steps": 1,
}
```

**To expand to 5 steps after 3-step success:**
```python
"num_steps": 5,  # Change from 3 to 5
```

Then push again:
```bash
kaggle kernels push -p .
```

---

## Expected Execution Times

| Step | Duration | Notes |
|------|----------|-------|
| Repo clone/pull | 1–2 min | First run slower |
| Dependencies install | 2–5 min | PyTorch download is large |
| HF authentication | <1 min | Token validation |
| Model download | 5–10 min | 15 GB Bodhan weights |
| Smoke test (3 steps) | 2–5 min | Depends on GPU |
| **Total** | **15–25 min** | First run; subsequent runs faster |

---

## Output Files

After execution, Kaggle saves:

```
/kaggle/working/outputs/
└── kaggle_smoke_test_results.json
```

Download this file to review:
```bash
kaggle kernels output your-username/your-username-bodhan-smoke-test -p ./
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Ensure CUDA PyTorch is installed. The script should handle this, but if not:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem: "Timeout waiting for model download"

**Cause:** Network congestion or model server slow  
**Solution:**
1. Re-run the kernel (might get better connection)
2. Check Kaggle's status page for network issues
3. Try again in a few hours

### Problem: "CUDA out of memory"

**Expected during testing.** Record the exact VRAM used. This tells us memory requirements for full training.

**To reduce memory usage, modify smoke test config:**
```python
SMOKE_TEST_CONFIG = {
    "batch_size": 1,              # Already minimal
    "max_seq_length": 1024,       # Reduce from 2600
    "dtype": "float16",           # Use float16 instead of bf16
}
```

### Problem: Results file is empty or incomplete

**Cause:** Script crashed or was interrupted  
**Solution:**
1. Check kernel output for error messages
2. Verify GPU was enabled (Step 3.1)
3. Verify HF_TOKEN was added (Step 3.2)
4. Re-run the kernel

---

## Next Steps After Successful Smoke Test

If the 3-step smoke test passes:

1. **Expand to 5 steps:**
   - Change `num_steps` to 5 in `run_bodhan_smoke_test.py`
   - Push and re-run

2. **Review metrics:**
   - Check loss progression (should decrease)
   - Check GPU memory usage
   - Verify backward pass works

3. **Run full training:**
   - Modify script to use full dataset: `max_samples=None`
   - Set `num_steps` for multiple epochs
   - Run for 1–2 epochs as proof-of-concept

---

## Important Notes

- ❌ **Do NOT commit credentials, tokens, or secrets** to this directory
- ❌ **Do NOT commit model weights** (they're downloaded on-demand)
- ✅ The validated training formulation is **NOT modified** by this script
- ✅ This is a **read-only execution environment** for validation only
- ✅ All results are saved to the output file for download

---

## Questions or Issues?

If you encounter problems:
1. Check this README thoroughly
2. Review the kernel output logs (visible on Kaggle)
3. Check `kaggle_smoke_test_results.json` for error messages
4. Review the main `run_bodhan_smoke_test.py` script for debugging

---

**Setup Created:** 2026-09-22  
**Script Version:** 1.0  
**Status:** Ready for GPU execution on Kaggle
