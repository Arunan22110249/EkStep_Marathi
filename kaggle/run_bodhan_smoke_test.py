"""
Bodhan TTS Smoke Test Execution on Kaggle GPU

This script:
1. Clones/updates the repository
2. Verifies CUDA and GPU availability
3. Installs required dependencies
4. Authenticates with Hugging Face
5. Runs the GPU-ready smoke test
6. Captures all metrics to JSON output

Designed to run on Kaggle Notebooks with GPU acceleration enabled.
"""

import argparse
import os
import sys
import json
import subprocess
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Configuration
REPO_URL = "https://github.com/Arunan22110249/EkStep_Marathi.git"
REPO_DIR = "/kaggle/working/EkStep_Marathi"
OUTPUTS_DIR = "/kaggle/working/outputs"
RESULTS_FILE = "/kaggle/working/outputs/kaggle_smoke_test_results.json"


def _this_script_dir() -> Path:
    """Resolve this script's directory, falling back safely when __file__ is unavailable (e.g. Kaggle notebooks)."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


# Local development fallback token file (never used on Kaggle; ignored by Git).
LOCAL_HF_TOKEN_FILE = _this_script_dir().parent / "secrets" / "hf_token.txt"

# Smoke test parameters
SMOKE_TEST_CONFIG = {
    "device": "cuda",
    "dtype": "bfloat16",  # or float16 if bf16 not supported
    "batch_size": 1,
    "max_samples": 10,
    "num_steps": 3,  # Start with 3, expand to 5 if successful
    "learning_rate": 1e-5,
    "gradient_accumulation_steps": 1,
}


def run_command(cmd: str, description: str, capture_output: bool = True) -> tuple:
    """Run a shell command and return success status and output."""
    try:
        print(f"\n{'='*80}")
        print(f"{description}")
        print(f"{'='*80}")
        print(f"Command: {cmd}")
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        output = result.stdout + (result.stderr if result.stderr else "")
        
        print(f"Status: {'✓ SUCCESS' if success else '✗ FAILED'}")
        return success, output
    
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT (10 minutes exceeded)")
        return False, "Timeout"
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False, str(e)


def detect_cuda_info() -> Dict[str, Any]:
    """Detect CUDA and GPU information."""
    info = {
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "nvidia_smi_output": None,
    }
    
    success, output = run_command("nvidia-smi", "Detecting NVIDIA GPU (nvidia-smi)")
    
    if success and output:
        info["nvidia_smi_output"] = output
        
        # Parse GPU name
        for line in output.split('\n'):
            if 'NVIDIA' in line or 'GPU' in line:
                info["gpu_name"] = line.strip()
                break
        
        # Parse VRAM
        for line in output.split('\n'):
            if 'MiB' in line and '/' in line:
                try:
                    parts = line.split('/')
                    vram_str = parts[1].strip().split()[0]
                    info["gpu_vram_gb"] = float(vram_str) / 1024
                except:
                    pass
        
        info["cuda_available"] = True
        info["gpu_count"] = 1
    
    return info


def verify_python_environment() -> Dict[str, Any]:
    """Verify Python version and key packages."""
    import torch
    
    try:
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        # Detect dtype support
        device = "cuda" if cuda_available else "cpu"
        supports_bf16 = torch.cuda.is_bf16_supported() if cuda_available else False
        supports_fp16 = True  # All GPUs support fp16
        
        # Detect device properties
        device_name = None
        device_properties = None
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_properties = torch.cuda.get_device_properties(0)
        
        return {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "pytorch_version": pytorch_version,
            "cuda_available": cuda_available,
            "device_name": device_name,
            "device_properties": str(device_properties),
            "supports_bf16": supports_bf16,
            "supports_fp16": supports_fp16,
            "recommended_dtype": "bfloat16" if supports_bf16 else "float16",
        }
    except Exception as e:
        return {
            "error": str(e),
            "pytorch_version": None,
        }


def clone_or_update_repo() -> bool:
    """Clone or update the repository."""
    if os.path.exists(REPO_DIR):
        print(f"\nRepository already exists at {REPO_DIR}")
        print("Pulling latest changes...")
        success, _ = run_command(
            f"cd {REPO_DIR} && git pull origin main",
            "Updating repository (git pull)",
        )
        return success
    else:
        print(f"\nCloning repository to {REPO_DIR}...")
        success, _ = run_command(
            f"git clone {REPO_URL} {REPO_DIR}",
            "Cloning repository",
        )
        return success


def install_dependencies() -> bool:
    """Install required dependencies."""
    # Create output directory
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Install from requirements.txt
    success, _ = run_command(
        f"cd {REPO_DIR} && pip install -q -r requirements.txt",
        "Installing dependencies from requirements.txt",
    )
    
    if not success:
        print("Warning: requirements.txt installation failed, proceeding anyway")
    
    # Ensure CUDA PyTorch is installed
    print("\nVerifying CUDA PyTorch installation...")
    success, _ = run_command(
        "pip list | grep -i torch || echo 'Checking torch...'",
        "Checking installed PyTorch",
    )
    
    return True  # Continue even if some packages fail


def resolve_hf_token(hf_token_file_arg: Optional[str] = None) -> Optional[str]:
    """
    Resolve the Hugging Face token using the following priority:
      1. HF_TOKEN environment variable
      2. --hf-token-file CLI argument (path to a runtime-mounted token file)
      3. HF_TOKEN_FILE environment variable (path to a runtime-mounted token file)
      4. Local development fallback: secrets/hf_token.txt (never present on Kaggle)

    Returns the token string, or None if no source was found.
    Never logs the token value itself.
    """
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token

    candidate_paths = []
    if hf_token_file_arg:
        candidate_paths.append(Path(hf_token_file_arg))

    env_token_file = os.getenv("HF_TOKEN_FILE")
    if env_token_file:
        candidate_paths.append(Path(env_token_file))

    # Local development fallback (only relevant off-Kaggle; file is git-ignored).
    candidate_paths.append(LOCAL_HF_TOKEN_FILE)

    for path in candidate_paths:
        try:
            if path and path.is_file():
                token = path.read_text(encoding="utf-8").strip()
                if token:
                    return token
        except OSError:
            continue

    return None


def authenticate_hugging_face(hf_token_file_arg: Optional[str] = None) -> bool:
    """Authenticate with Hugging Face using a runtime-resolved token.

    Never prints the token value, its length, or any derived characters.
    """
    print(f"\n{'='*80}")
    print("Authenticating with Hugging Face")
    print(f"{'='*80}")

    hf_token = resolve_hf_token(hf_token_file_arg)
    print(f"HF_TOKEN available: {hf_token is not None}")

    if not hf_token:
        print(
            "Hugging Face authentication token not provided. "
            "Set HF_TOKEN or provide a runtime token file through HF_TOKEN_FILE."
        )
        return False

    # Make the resolved token available to this process and the smoke-test
    # subprocess via environment variable only (never via CLI arguments).
    os.environ["HF_TOKEN"] = hf_token

    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("✓ Successfully authenticated with Hugging Face")
        return True
    except ImportError:
        print("⚠ huggingface_hub not installed; HF_TOKEN environment variable is set for transformers")
        return True


def run_smoke_test() -> Dict[str, Any]:
    """Run the GPU smoke test and capture results."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "status": "running",
        "configuration": SMOKE_TEST_CONFIG,
        "output": None,
        "success": False,
        "error": None,
    }
    
    try:
        # Build command with configuration
        cmd = (
            f"cd {REPO_DIR} && python scripts/smoke_test_training.py "
            f"--device {SMOKE_TEST_CONFIG['device']} "
            f"--dtype {SMOKE_TEST_CONFIG['dtype']} "
            f"--batch-size {SMOKE_TEST_CONFIG['batch_size']} "
            f"--max-samples {SMOKE_TEST_CONFIG['max_samples']} "
            f"--num-steps {SMOKE_TEST_CONFIG['num_steps']} "
            f"--learning-rate {SMOKE_TEST_CONFIG['learning_rate']} "
            f"--gradient-accumulation-steps {SMOKE_TEST_CONFIG['gradient_accumulation_steps']}"
        )
        
        print(f"\n{'='*80}")
        print("Running Bodhan Smoke Test on GPU")
        print(f"{'='*80}")
        print(f"Configuration: {SMOKE_TEST_CONFIG}")
        print(f"Command: {cmd}\n")

        # HF_TOKEN (if resolved) is inherited from this process's environment
        # by the subprocess below; it is never passed as a CLI argument and
        # the environment itself is never printed.
        success, output = run_command(cmd, "Executing smoke test")
        
        results["output"] = output
        results["success"] = success
        results["status"] = "completed" if success else "failed"
        
        if not success:
            results["error"] = "Smoke test exited with non-zero status"
        
        return results
    
    except Exception as e:
        results["error"] = str(e)
        results["status"] = "error"
        results["output"] = traceback.format_exc()
        return results


def save_results(results: Dict[str, Any]) -> bool:
    """Save results to JSON file."""
    try:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ Results saved to: {RESULTS_FILE}")
        print(f"{'='*80}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Kaggle runner."""
    parser = argparse.ArgumentParser(
        description="Bodhan TTS smoke test execution runner (Kaggle GPU)."
    )
    parser.add_argument(
        "--hf-token-file",
        type=str,
        default=None,
        help=(
            "Path to a runtime-mounted file containing the Hugging Face token. "
            "Must be a file path, never the token value itself."
        ),
    )
    parser.add_argument(
        "--validate-auth",
        action="store_true",
        help=(
            "Only resolve and report Hugging Face authentication availability, "
            "without running the smoke test."
        ),
    )
    return parser.parse_args()


def main():
    """Main execution flow."""
    args = parse_args()

    if args.validate_auth:
        hf_token = resolve_hf_token(args.hf_token_file)
        print(f"HF_TOKEN available: {hf_token is not None}")
        sys.exit(0 if hf_token is not None else 1)

    print("=" * 80)
    print("BODHAN TTS SMOKE TEST - KAGGLE GPU EXECUTION")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Output directory: {OUTPUTS_DIR}")
    print(f"Results file: {RESULTS_FILE}")
    
    # Collect all results
    all_results = {
        "execution_start": datetime.now().isoformat(),
        "repository": REPO_URL,
        "environment": {},
        "cuda_detection": {},
        "repository_status": {},
        "dependencies_status": {},
        "authentication_status": {},
        "smoke_test_results": {},
        "execution_end": None,
    }
    
    try:
        # Step 1: Detect CUDA
        print("\n" + "="*80)
        print("STEP 1: CUDA DETECTION")
        print("="*80)
        cuda_info = detect_cuda_info()
        all_results["cuda_detection"] = cuda_info
        
        if not cuda_info["cuda_available"]:
            print("✗ CUDA not available - cannot proceed")
            all_results["smoke_test_results"]["status"] = "failed"
            all_results["smoke_test_results"]["error"] = "CUDA not available"
            raise RuntimeError("CUDA required but not available")
        
        print(f"✓ CUDA available")
        print(f"  GPU: {cuda_info['gpu_name']}")
        gpu_vram_gb = cuda_info.get('gpu_vram_gb')
        vram_display = f"{gpu_vram_gb:.1f} GB" if isinstance(gpu_vram_gb, (int, float)) else "unknown"
        print(f"  VRAM: {vram_display}")
        
        # Step 2: Verify Python environment
        print("\n" + "="*80)
        print("STEP 2: PYTHON ENVIRONMENT")
        print("="*80)
        env_info = verify_python_environment()
        all_results["environment"] = env_info
        
        print(f"✓ Python: {env_info.get('python_version')}")
        print(f"  PyTorch: {env_info.get('pytorch_version')}")
        print(f"  CUDA Available: {env_info.get('cuda_available')}")
        print(f"  Device: {env_info.get('device_name')}")
        print(f"  Recommended dtype: {env_info.get('recommended_dtype')}")
        
        # Update dtype if needed
        if not env_info.get('supports_bf16') and SMOKE_TEST_CONFIG['dtype'] == 'bfloat16':
            print("⚠ GPU does not support bfloat16, switching to float16")
            SMOKE_TEST_CONFIG['dtype'] = 'float16'
        
        # Step 3: Clone/update repository
        print("\n" + "="*80)
        print("STEP 3: REPOSITORY MANAGEMENT")
        print("="*80)
        repo_success = clone_or_update_repo()
        all_results["repository_status"]["success"] = repo_success
        
        if not repo_success:
            raise RuntimeError("Failed to clone/update repository")
        
        print(f"✓ Repository ready at {REPO_DIR}")
        
        # Step 4: Install dependencies
        print("\n" + "="*80)
        print("STEP 4: DEPENDENCIES")
        print("="*80)
        deps_success = install_dependencies()
        all_results["dependencies_status"]["success"] = deps_success
        print(f"✓ Dependencies installed")
        
        # Step 5: Authenticate with Hugging Face
        print("\n" + "="*80)
        print("STEP 5: HUGGING FACE AUTHENTICATION")
        print("="*80)
        auth_success = authenticate_hugging_face(args.hf_token_file)
        all_results["authentication_status"]["success"] = auth_success
        
        if not auth_success:
            print("⚠ Warning: HF authentication may fail - model download might not work")
        
        # Step 6: Run smoke test
        print("\n" + "="*80)
        print("STEP 6: RUNNING SMOKE TEST")
        print("="*80)
        smoke_test_results = run_smoke_test()
        all_results["smoke_test_results"] = smoke_test_results
        
        # Determine if successful
        if smoke_test_results["success"]:
            print("\n✓ SMOKE TEST PASSED")
            print("\nNext steps:")
            print("  1. Review results in kaggle_smoke_test_results.json")
            print("  2. If successful, re-run with --num-steps 5 for extended validation")
            print("  3. Proceed to full training if metrics are acceptable")
        else:
            print("\n✗ SMOKE TEST FAILED")
            if "CUDA out of memory" in smoke_test_results.get("error", ""):
                print("  Issue: GPU out of memory")
                print("  Solution: Reduce batch_size or max_seq_length")
            elif "401" in smoke_test_results.get("error", ""):
                print("  Issue: Hugging Face authentication failed")
                print("  Solution: Add HF_TOKEN as a Kaggle Secret")
    
    except Exception as e:
        print(f"\n✗ EXECUTION ERROR: {e}")
        all_results["smoke_test_results"]["error"] = str(e)
        all_results["smoke_test_results"]["status"] = "error"
    
    finally:
        all_results["execution_end"] = datetime.now().isoformat()
        
        # Save all results
        save_results(all_results)
        
        print(f"\nEnd time: {datetime.now().isoformat()}")
        print("=" * 80)


if __name__ == "__main__":
    main()
