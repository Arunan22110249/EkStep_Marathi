"""
Lightweight authenticated inspection of bodhan-ai/indic-speak repository.
Downloads only small text/config files without downloading model weights.
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional


MODEL_ID = "bodhan-ai/indic-speak"
SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "token_contract.md",
    "inference.py",
    "voices.md",
]


def safe_imports() -> tuple[dict, dict]:
    versions = {
        "transformers": "MISSING",
        "huggingface_hub": "MISSING",
        "snac": "MISSING",
        "torch": "MISSING",
    }
    errors = {}

    for pkg in ["transformers", "huggingface_hub", "snac", "torch"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            errors[pkg] = f"{type(exc).__name__}: {exc}"

    return versions, errors


def download_small_file(filename: str) -> tuple[Optional[str], Optional[str]]:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=MODEL_ID,
            repo_type="model",
            filename=filename,
            local_dir=None,
            force_download=False,
            resume_download=True,
        )
        return path, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:150]}"


def read_json_file(filepath: str) -> tuple[Optional[dict], Optional[str]]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_text_file(filepath: str, max_lines: int = 100) -> tuple[Optional[str], Optional[str]]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        content = "".join(lines[:max_lines])
        if len(lines) > max_lines:
            content += f"\n... ({len(lines) - max_lines} more lines)"
        return content, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def inspect_config(config_data: dict) -> dict:
    result = {}
    if config_data:
        result["model_type"] = config_data.get("model_type", "UNKNOWN")
        result["architectures"] = config_data.get("architectures", [])
        result["hidden_size"] = config_data.get("hidden_size", "UNKNOWN")
        result["num_hidden_layers"] = config_data.get("num_hidden_layers", "UNKNOWN")
        result["num_attention_heads"] = config_data.get("num_attention_heads", "UNKNOWN")
        result["vocab_size"] = config_data.get("vocab_size", "UNKNOWN")
        result["torch_dtype"] = config_data.get("torch_dtype", "UNKNOWN")
        result["auto_map"] = config_data.get("auto_map", {})
        result["additional_keys"] = sorted([k for k in config_data.keys() if k not in [
            "model_type", "architectures", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "vocab_size", "torch_dtype", "auto_map"
        ]])
    return result


def inspect_tokenizer_config(config_data: dict) -> dict:
    result = {}
    if config_data:
        result["tokenizer_class"] = config_data.get("tokenizer_class", "UNKNOWN")
        result["model_max_length"] = config_data.get("model_max_length", "UNKNOWN")
        result["special_tokens"] = config_data.get("special_tokens", {})
        result["additional_keys"] = sorted([k for k in config_data.keys() if k not in [
            "tokenizer_class", "model_max_length", "special_tokens"
        ]])
    return result


def print_section(title: str, content: Any = None):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    if content is not None:
        if isinstance(content, dict):
            for k, v in content.items():
                print(f"  {k}: {v}")
        elif isinstance(content, list):
            for item in content:
                print(f"  - {item}")
        else:
            print(str(content))


def main():
    versions, import_errors = safe_imports()
    
    print_section("Environment & Dependencies")
    for pkg, ver in versions.items():
        print(f"  {pkg}: {ver}")
    
    if import_errors:
        print("\n  Warnings:")
        for pkg, err in import_errors.items():
            print(f"    {pkg}: {err}")
    
    missing_hf = versions["huggingface_hub"] == "MISSING"
    if missing_hf:
        print_section("ERROR")
        print("  huggingface_hub is not installed or importable.")
        print("  Cannot proceed with repository inspection without Hugging Face Hub API.")
        return 1
    
    print_section(f"Repository: {MODEL_ID}")
    print(f"  Repository access check...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_info = api.model_info(repo_id=MODEL_ID, files_metadata=True)
        print(f"  ✓ Repository accessible")
        print(f"  Files in repository: ~{len(repo_info.siblings) if repo_info.siblings else '?'}")
    except Exception as exc:
        print_section("ERROR")
        print(f"  Repository access failed: {type(exc).__name__}: {str(exc)[:200]}")
        return 1
    
    print_section("Downloading Small Configuration Files")
    downloaded_files = {}
    
    for filename in SMALL_FILES:
        print(f"\n  {filename}...", end=" ", flush=True)
        filepath, error = download_small_file(filename)
        if filepath:
            print(f"✓ ({Path(filepath).stat().st_size} bytes)")
            downloaded_files[filename] = filepath
        else:
            print(f"✗ {error}")
    
    print("\n" + "="*70)
    print("File Inspection Results")
    print("="*70)
    
    if "config.json" in downloaded_files:
        config_data, err = read_json_file(downloaded_files["config.json"])
        if err:
            print(f"\nconfig.json: ERROR ({err})")
        else:
            config_info = inspect_config(config_data)
            print("\nconfig.json:")
            for k, v in config_info.items():
                if k == "additional_keys" and v:
                    print(f"  {k}: {', '.join(v[:5])}" + ("..." if len(v) > 5 else ""))
                elif k != "additional_keys":
                    print(f"  {k}: {v}")
    
    if "generation_config.json" in downloaded_files:
        gen_config, err = read_json_file(downloaded_files["generation_config.json"])
        if err:
            print(f"\ngeneration_config.json: ERROR ({err})")
        else:
            print("\ngeneration_config.json:")
            if gen_config:
                for k in sorted(gen_config.keys())[:10]:
                    v = gen_config[k]
                    if isinstance(v, str) and len(v) > 60:
                        v = v[:60] + "..."
                    print(f"  {k}: {v}")
                if len(gen_config) > 10:
                    print(f"  ... and {len(gen_config) - 10} more keys")
    
    if "tokenizer_config.json" in downloaded_files:
        tok_config, err = read_json_file(downloaded_files["tokenizer_config.json"])
        if err:
            print(f"\ntokenizer_config.json: ERROR ({err})")
        else:
            tok_info = inspect_tokenizer_config(tok_config)
            print("\ntokenizer_config.json:")
            for k, v in tok_info.items():
                if k == "special_tokens" and isinstance(v, dict):
                    print(f"  {k}:")
                    for sk in sorted(v.keys())[:5]:
                        print(f"    {sk}: {v[sk]}")
                    if len(v) > 5:
                        print(f"    ... and {len(v) - 5} more")
                elif k == "additional_keys" and v:
                    print(f"  {k}: {', '.join(v[:5])}" + ("..." if len(v) > 5 else ""))
                elif k != "additional_keys":
                    print(f"  {k}: {v}")
    
    if "token_contract.md" in downloaded_files:
        content, err = read_text_file(downloaded_files["token_contract.md"], max_lines=50)
        if err:
            print(f"\ntoken_contract.md: ERROR ({err})")
        else:
            print("\ntoken_contract.md: (first 50 lines)")
            if content:
                for line in content.split("\n")[:50]:
                    print(f"  {line}")
    
    if "inference.py" in downloaded_files:
        content, err = read_text_file(downloaded_files["inference.py"], max_lines=50)
        if err:
            print(f"\ninference.py: ERROR ({err})")
        else:
            print("\ninference.py: (first 50 lines)")
            if content:
                for line in content.split("\n")[:50]:
                    print(f"  {line}")
    
    if "voices.md" in downloaded_files:
        content, err = read_text_file(downloaded_files["voices.md"], max_lines=100)
        if err:
            print(f"\nvoices.md: ERROR ({err})")
        else:
            print("\nvoices.md: (content)")
            if content:
                for line in content.split("\n"):
                    print(f"  {line}")
    
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    print(f"  Small files downloaded: {len(downloaded_files)}/{len(SMALL_FILES)}")
    print(f"  model.safetensors downloaded: NO (intentionally skipped)")
    print(f"  Datasets downloaded: NO")
    print(f"  Credentials exposed: NO")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
