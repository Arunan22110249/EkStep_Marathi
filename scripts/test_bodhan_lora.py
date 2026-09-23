"""
Test PEFT LoRA attachment to the Bodhan architecture without loading
the pretrained checkpoint weights.

This validates:
1. Bodhan config can construct the architecture.
2. PEFT can attach LoRA to the actual Llama projection modules.
3. The expected trainable parameters are created.
4. The adapter can be saved and reloaded.

No pretrained 3.3B/3.78B checkpoint weights are downloaded.
"""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


MODEL_ID = "bodhan-ai/indic-speak"
OUTPUT_DIR = Path("outputs/lora_smoke_test")

# Attention-only LoRA keeps the first validation deliberately small.
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


def main():
    print("=" * 80)
    print("BODHAN LoRA ATTACHMENT TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load configuration only
    # ------------------------------------------------------------------
    print("\n[1/6] Loading Bodhan configuration...")

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    print(f"Model type : {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Layers     : {config.num_hidden_layers}")
    print(f"Vocab size : {config.vocab_size}")

    # ------------------------------------------------------------------
    # 2. Construct architecture without pretrained weights
    # ------------------------------------------------------------------
    print("\n[2/6] Constructing model from config...")

    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
    )

    print("Architecture constructed successfully.")

    # ------------------------------------------------------------------
    # 3. Configure LoRA
    # ------------------------------------------------------------------
    print("\n[3/6] Creating LoRA configuration...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    print("Target modules:")
    for module in TARGET_MODULES:
        print(f"  - {module}")

    # ------------------------------------------------------------------
    # 4. Attach LoRA
    # ------------------------------------------------------------------
    print("\n[4/6] Attaching LoRA...")

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(
        f"Trainable percentage : "
        f"{100.0 * trainable / total:.4f}%"
    )

    # ------------------------------------------------------------------
    # 5. Verify LoRA modules actually exist
    # ------------------------------------------------------------------
    print("\n[5/6] Verifying LoRA modules...")

    lora_parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if "lora_" in name
    ]

    if not lora_parameter_names:
        raise RuntimeError(
            "No LoRA parameters were found after PEFT attachment."
        )

    print(f"Found {len(lora_parameter_names)} LoRA parameter tensors.")

    for name in lora_parameter_names[:12]:
        print(f"  {name}")

    if len(lora_parameter_names) > 12:
        print(
            f"  ... and {len(lora_parameter_names) - 12} more"
        )

    # ------------------------------------------------------------------
    # 6. Save adapter
    # ------------------------------------------------------------------
    print("\n[6/6] Saving LoRA adapter...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(OUTPUT_DIR)

    print(f"Adapter saved to: {OUTPUT_DIR.resolve()}")

    # Verify that PEFT can reload the adapter into the same architecture.
    print("\nReloading saved adapter...")

    base_model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
    )

    reloaded = PeftModel.from_pretrained(
        base_model,
        OUTPUT_DIR,
    )

    reloaded_trainable = sum(
        p.numel()
        for p in reloaded.parameters()
        if p.requires_grad
    )

    print(
        f"Reloaded trainable parameters: "
        f"{reloaded_trainable:,}"
    )

    print("\n" + "=" * 80)
    print("BODHAN LoRA ATTACHMENT TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()