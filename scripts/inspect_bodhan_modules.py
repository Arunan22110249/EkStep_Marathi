"""
Inspect Bodhan module names without downloading/loading model weights.
"""

from transformers import AutoConfig, AutoModelForCausalLM


MODEL_ID = "bodhan-ai/indic-speak"


def main():
    print("=" * 80)
    print("BODHAN MODULE INSPECTION")
    print("=" * 80)

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    print(f"Model type : {config.model_type}")
    print(f"Architecture: {config.architectures}")
    print()

    print("Constructing model from config only...")
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
    )

    print("Model constructed without loading checkpoint weights.")
    print()

    print("Attention / MLP Linear Modules")
    print("-" * 80)

    seen = set()

    for name, module in model.named_modules():
        class_name = module.__class__.__name__

        if class_name == "Linear":
            if any(
                keyword in name
                for keyword in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            ):
                if name not in seen:
                    print(f"{name:70} {class_name}")
                    seen.add(name)

    print()
    print("Candidate LoRA target module names")
    print("-" * 80)

    targets = set()

    for name, module in model.named_modules():
        if module.__class__.__name__ == "Linear":
            short_name = name.split(".")[-1]

            if short_name in {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            }:
                targets.add(short_name)

    for target in sorted(targets):
        print(target)


if __name__ == "__main__":
    main()