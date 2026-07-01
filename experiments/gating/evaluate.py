"""Evaluate the gated CoreModel.

Four metric groups:
  1. LM quality        - test perplexity.
  2. Router health     - per-mode utilization, entropy, balance CV, DONE rate.
  3. Mode alignment    - NMI / purity / confusion matrix vs held-out segments
                         (the key eval: did latent routing rediscover the modes?)
  4. Downstream        - tool-call structural validity, reasoning-answer match.

Usage:
    python -m experiments.gating.evaluate --checkpoint experiments/gating/checkpoints/core_model.pt
"""

import argparse
import json
import math
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.gating.config import ExperimentConfig, SEGMENT_NAMES, build_model_config
from experiments.gating.data import (
    MixedModeDataset,
    build_mixed_examples,
    collate_mixed,
    split_examples,
)
from experiments.gating.metrics import mode_alignment, router_health
from src.models.core_model import CoreModelForCausalLM


@torch.no_grad()
def evaluate(model, loader, num_modes: int, done_mode_id: int):
    model.eval()
    total_lm, total_tok = 0.0, 0
    gate_accum, n_gate = None, 0
    all_pred, all_seg = [], []
    done_fires, n_seq = 0, 0

    for batch in loader:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        ntok = int((batch["labels"] != -100).sum())
        total_lm += float(out["lm_loss"]) * ntok
        total_tok += ntok

        gw = out["gate_weights"]
        if gw is None:
            continue
        # Router health.
        h = router_health(gw, batch["attention_mask"])
        gate_accum = h if gate_accum is None else {k: gate_accum[k] + h[k] for k in h}
        n_gate += 1

        # Per-position argmax mode (avg over gate layers).
        pred_modes = gw.mean(dim=0).argmax(dim=-1)  # [B, T]

        # Mode alignment: keep positions with a real segment label.
        seg = batch["segment_ids"]
        valid = seg != -100
        all_pred.append(pred_modes[valid])
        all_seg.append(seg[valid])

        # DONE fire rate: does DONE become the top mode anywhere in each sequence?
        done_top = (pred_modes == done_mode_id) & (batch["attention_mask"].bool())
        done_fires += int(done_top.any(dim=1).sum())
        n_seq += batch["attention_mask"].shape[0]

    results = {
        "test_lm_loss": total_lm / max(total_tok, 1),
        "test_ppl": math.exp(total_lm / max(total_tok, 1)),
        "done_fire_rate_per_seq": done_fires / max(n_seq, 1),
    }
    if n_gate:
        results.update({k: v / n_gate for k, v in gate_accum.items()})
    if all_pred:
        pred = torch.cat(all_pred)
        seg = torch.cat(all_seg)
        align = mode_alignment(pred, seg, num_modes=num_modes)
        results["mode_alignment"] = align
    return results


def pretty_confusion(conf, names):
    header = "true\\pred  " + "  ".join(f"{n:>8}" for n in names)
    lines = [header]
    for i, row in enumerate(conf):
        lines.append(f"{names[i]:>9}  " + "  ".join(f"{c:>8}" for c in row))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/gating/checkpoints/core_model.pt")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp = ExperimentConfig()
    if args.smoke:
        exp.num_hidden_layers = 4
        exp.hidden_size = 128
        exp.intermediate_size = 384
        exp.num_attention_heads = 4
        exp.num_key_value_heads = 4
        exp.gate_layer_indices = [2]
        exp.max_position_embeddings = 512
        exp.max_length = 512
        exp.max_samples_per_source = 32

    tokenizer = AutoTokenizer.from_pretrained(exp.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = build_model_config(exp)
    model = CoreModelForCausalLM(model_config).to(device)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint from {args.checkpoint} (step {ckpt.get('step')})")
    except FileNotFoundError:
        print(f"[WARN] no checkpoint at {args.checkpoint}; evaluating a randomly-initialized model.")

    examples = build_mixed_examples(
        tokenizer, exp.reasoning_csv, exp.tool_csv, exp.plain_csv,
        exp.max_length, exp.max_samples_per_source,
    )
    _, _, test = split_examples(examples, exp.val_fraction, exp.test_fraction, exp.seed)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    loader = DataLoader(
        MixedModeDataset(test), batch_size=exp.batch_size, shuffle=False,
        collate_fn=partial(collate_mixed, pad_id=pad_id, device=device),
    )

    results = evaluate(model, loader, model_config.num_modes, model_config.done_mode_id)

    print("\n===== Evaluation =====")
    align = results.pop("mode_alignment", None)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()}, indent=2))
    if align:
        print(f"\nMode alignment: NMI={align['nmi']:.4f}  purity={align['purity']:.4f}")
        print("Confusion matrix (rows=true segment, cols=predicted mode):")
        print(pretty_confusion(align["confusion_matrix"], SEGMENT_NAMES))
    return results


if __name__ == "__main__":
    main()
