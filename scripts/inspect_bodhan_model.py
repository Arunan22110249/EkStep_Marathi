"""
Inspect Bodhan's model configuration without loading model weights.

This is intentionally configuration-only because the local environment
does not have CUDA and the Bodhan model is ~3.3B parameters.
"""

from transformers import AutoConfig


MODEL_ID = "bodhan-ai/indic-speak"


def main():
    print("=" * 70)
    print("BODHAN MODEL CONFIGURATION")
    print("=" * 70)

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    print(f"model_type          : {config.model_type}")
    print(f"architectures       : {config.architectures}")
    print(f"vocab_size          : {config.vocab_size}")
    print(f"hidden_size         : {config.hidden_size}")
    print(f"intermediate_size   : {config.intermediate_size}")
    print(f"num_hidden_layers   : {config.num_hidden_layers}")
    print(f"num_attention_heads : {config.num_attention_heads}")
    print(f"num_key_value_heads : {getattr(config, 'num_key_value_heads', None)}")
    print(f"hidden_act          : {config.hidden_act}")
    print(f"max_position_emb    : {config.max_position_embeddings}")
    print(f"rope_theta          : {getattr(config, 'rope_theta', None)}")
    print()

    print("LoRA-relevant architecture")
    print("-" * 70)

    for attr in [
        "attention_bias",
        "mlp_bias",
        "pretraining_tp",
        "tie_word_embeddings",
    ]:
        print(f"{attr:20}: {getattr(config, attr, None)}")

    print()
    print("All configuration keys containing attention/MLP information:")
    print("-" * 70)

    for key in sorted(config.to_dict()):
        key_lower = key.lower()
        if any(
            term in key_lower
            for term in ["attn", "attention", "mlp", "proj", "q_proj", "k_proj", "v_proj"]
        ):
            print(f"{key}: {getattr(config, key, None)}")


if __name__ == "__main__":
    main()