"""Transfer pretrained nyx_reasoning weights into the gated CoreModel.

The gated ``CoreModelForCausalLM`` reuses the exact ``CoreOutlineDecoderLayer``
backbone, so its parameter keys are a *superset* of the nyx_reasoning
checkpoint: every embedding / attention / MLP / norm / lm_head tensor matches
1:1, and only the small gating parameters (``model.gates.*``) are new. We
therefore do a **name + shape matched copy**: each source tensor is copied into
the target where the key exists and the shape agrees; anything left over (the
gate) keeps its fresh initialization. This lets training start from the
pretrained reasoning model instead of from scratch.

Usage:
    python -m scripts.transfer_weights \
        --src "models/nyx_reasoning (2).pth" \
        --dst "models/nyx_gated_agentic.pth"

Optional gating flags mirror CoreModelConfig (e.g. --gate-layers 12).
The script is memory-conscious: it frees the source state dict before saving.
"""

import argparse
import json
import os
from typing import Dict

import torch

from src.models.core_model import CoreModelConfig, CoreModelForCausalLM


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a checkpoint that may be a raw state_dict or a wrapper dict."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj  # already a raw state_dict


def infer_config(sd: Dict[str, torch.Tensor], args) -> CoreModelConfig:
    """Infer the backbone architecture directly from checkpoint tensor shapes."""
    vocab_size, hidden_size = sd["model.embed_tokens.weight"].shape

    layer_ids = [
        int(k.split(".")[2]) for k in sd if k.startswith("model.layers.")
    ]
    num_layers = max(layer_ids) + 1

    # head_dim from a rotary inv_freq buffer (length = head_dim / 2).
    inv_freq = sd.get("model.layers.0.self_attn.rotary_emb.inv_freq")
    head_dim = 2 * inv_freq.shape[0] if inv_freq is not None else 64
    num_heads = hidden_size // head_dim

    kv_dim = sd["model.layers.0.self_attn.k_proj.weight"].shape[0]
    num_kv_heads = max(kv_dim // head_dim, 1)

    intermediate = sd["model.layers.0.mlp.gate_proj.weight"].shape[0]

    gate_layers = args.gate_layers if args.gate_layers else [num_layers // 2]

    print(
        f"Inferred config: vocab={vocab_size} hidden={hidden_size} "
        f"layers={num_layers} heads={num_heads} kv_heads={num_kv_heads} "
        f"head_dim={head_dim} intermediate={intermediate}"
    )

    return CoreModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=args.max_position_embeddings,
        gate_layer_indices=gate_layers,
        share_gate=not args.independent_gates,
        gate_noise_std=args.gate_noise_std,
        mode_balance_target=[0.33, 0.30, 0.33, 0.04],
    )


@torch.no_grad()
def copy_matching(src_sd: Dict[str, torch.Tensor], model: torch.nn.Module):
    """Copy every source tensor whose key+shape match the target; report the rest."""
    tgt_sd = model.state_dict()
    copied, shape_mismatch, unexpected = [], [], []

    for k, v in src_sd.items():
        if k not in tgt_sd:
            unexpected.append(k)
        elif tgt_sd[k].shape != v.shape:
            shape_mismatch.append((k, tuple(v.shape), tuple(tgt_sd[k].shape)))
        else:
            tgt_sd[k] = v.clone()
            copied.append(k)

    model.load_state_dict(tgt_sd, strict=True)
    missing = [k for k in tgt_sd if k not in copied]
    return copied, missing, shape_mismatch, unexpected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="models/nyx_reasoning (2).pth")
    parser.add_argument("--dst", default="models/nyx_gated_agentic.pth")
    parser.add_argument(
        "--gate-layers", type=int, nargs="*", default=None,
        help="Decoder layer indices to gate (default: [num_layers//2]).",
    )
    parser.add_argument("--independent-gates", action="store_true",
                        help="Use one gate per gate layer instead of a shared gate.")
    parser.add_argument("--gate-noise-std", type=float, default=0.3)
    parser.add_argument("--max-position-embeddings", type=int, default=32768)
    parser.add_argument("--save-config", action="store_true", default=True,
                        help="Also write <dst>.config.json.")
    args = parser.parse_args()

    print(f"Loading source checkpoint: {args.src}")
    src_sd = load_state_dict(args.src)
    print(f"Source tensors: {len(src_sd)}")

    config = infer_config(src_sd, args)

    print("Building gated CoreModel and copying matching weights...")
    model = CoreModelForCausalLM(config)
    copied, missing, shape_mismatch, unexpected = copy_matching(src_sd, model)

    del src_sd  # free ~2.4GB before saving

    print("\n===== Transfer report =====")
    print(f"Copied      : {len(copied)} tensors")
    print(f"Missing     : {len(missing)} tensors (kept fresh init)")
    for k in missing:
        print(f"    + {k}")
    print(f"Shape-skip  : {len(shape_mismatch)}")
    for k, s, t in shape_mismatch:
        print(f"    ! {k}: src{s} vs tgt{t}")
    print(f"Unexpected  : {len(unexpected)} (in src, not in target)")
    for k in unexpected[:20]:
        print(f"    - {k}")

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.dst)
    print(f"\nSaved gated model weights -> {args.dst}")

    if args.save_config:
        cfg_path = args.dst + ".config.json"
        cfg_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2, default=str)
        print(f"Saved config -> {cfg_path}")

    # Sanity: the only expected 'missing' keys are gate params.
    non_gate_missing = [k for k in missing if ".gates." not in k]
    if non_gate_missing:
        print(f"\n[WARN] {len(non_gate_missing)} non-gate tensors were NOT loaded:")
        for k in non_gate_missing:
            print(f"    {k}")
    else:
        print("\n[OK] All backbone/head weights transferred; only the gate is new.")


if __name__ == "__main__":
    main()
