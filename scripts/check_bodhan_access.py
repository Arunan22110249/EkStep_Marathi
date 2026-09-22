import os
import sys
from typing import Optional


MODEL_ID = "bodhan-ai/indic-speak"
REQUIRED_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json"]


def safe_imports():
    versions = {"transformers": "MISSING", "huggingface_hub": "MISSING"}
    errors = {}

    try:
        import transformers
        versions["transformers"] = getattr(transformers, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic only
        errors["transformers"] = f"{type(exc).__name__}: {exc}"

    try:
        import huggingface_hub
        versions["huggingface_hub"] = getattr(huggingface_hub, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic only
        errors["huggingface_hub"] = f"{type(exc).__name__}: {exc}"

    return versions, errors


def get_repo_files() -> tuple[Optional[list[str]], Optional[str]]:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id=MODEL_ID, repo_type="model")
        return files, None
    except Exception as exc:  # pragma: no cover - diagnostic only
        return None, f"{type(exc).__name__}: {exc}"


def download_if_present(filename: str):
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
        return True, path
    except Exception as exc:  # pragma: no cover - diagnostic only
        return False, f"{type(exc).__name__}: {exc}"


def print_report(report):
    print("=== Bodhan Access Test ===")
    print(f"Repository access: {report['repository_access']}")
    print(f"Config: {report['config']}")
    print(f"Tokenizer: {report['tokenizer']}")
    print(f"Tokenizer config: {report['tokenizer_config']}")
    print(f"File listing: {report['file_listing']}")
    print(f"Transformers: {report['transformers_version']}")
    print(f"Hugging Face Hub: {report['hub_version']}")
    print(f"Model weights visible: {report['model_weights_visible']}")
    print(f"Full model downloaded: NO")
    print(f"Next blocker: {report['next_blocker']}")


def main():
    versions, import_errors = safe_imports()
    report = {
        "repository_access": "UNKNOWN",
        "config": "UNKNOWN",
        "tokenizer": "UNKNOWN",
        "tokenizer_config": "UNKNOWN",
        "file_listing": "UNKNOWN",
        "transformers_version": versions["transformers"],
        "hub_version": versions["huggingface_hub"],
        "model_weights_visible": "UNKNOWN",
        "next_blocker": "None",
    }

    if import_errors.get("transformers"):
        report["next_blocker"] = "Install transformers or use the project Conda environment with Hugging Face dependencies."
    if import_errors.get("huggingface_hub"):
        report["next_blocker"] = "Install huggingface_hub or use the project Conda environment with Hugging Face dependencies."

    if "MISSING" not in (versions["transformers"], versions["huggingface_hub"]):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_access = api.model_info(repo_id=MODEL_ID, files_metadata=True)
            report["repository_access"] = "YES"
        except Exception as exc:  # pragma: no cover - diagnostic only
            report["repository_access"] = f"NO ({type(exc).__name__}: {exc})"
            report["next_blocker"] = f"Repository access failed: {type(exc).__name__}: {exc}"

    if report["repository_access"] == "YES":
        try:
            files, repo_files_err = get_repo_files()
            if files is not None:
                report["file_listing"] = "YES"
                item_names = [str(item) for item in files]
                if "model.safetensors" in item_names:
                    report["model_weights_visible"] = "YES"
                else:
                    report["model_weights_visible"] = "NO"
            else:
                report["file_listing"] = f"NO ({repo_files_err})"
                report["next_blocker"] = f"File listing failed: {repo_files_err}"
        except Exception as exc:  # pragma: no cover - diagnostic only
            report["file_listing"] = f"ERROR ({type(exc).__name__}: {exc})"
            report["next_blocker"] = f"File listing failed: {type(exc).__name__}: {exc}"

        for filename, key in [("config.json", "config"), ("tokenizer.json", "tokenizer"), ("tokenizer_config.json", "tokenizer_config")]:
            ok, detail = download_if_present(filename)
            if ok:
                report[key] = f"YES ({detail})"
            else:
                report[key] = f"NO ({detail})"
                if report["next_blocker"] == "None":
                    report["next_blocker"] = f"{filename} access failed: {detail}"

    else:
        report["file_listing"] = "SKIPPED (repo access failed)"
        report["config"] = "SKIPPED (repo access failed)"
        report["tokenizer"] = "SKIPPED (repo access failed)"
        report["tokenizer_config"] = "SKIPPED (repo access failed)"
        if report["model_weights_visible"] == "UNKNOWN":
            report["model_weights_visible"] = "NO"

    if report["repository_access"] == "YES" and report["model_weights_visible"] == "UNKNOWN":
        report["model_weights_visible"] = "NO"

    if report["next_blocker"] == "None" and report["repository_access"] == "YES":
        report["next_blocker"] = "No blocker from lightweight access checks; full model download remains intentionally disabled."

    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
