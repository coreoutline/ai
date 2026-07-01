"""Train the gated CoreModel with latent routing + aux-loss ramp.

Usage:
    python -m experiments.gating.train                 # full run (default config)
    python -m experiments.gating.train --smoke         # tiny/fast sanity run

The loss is  L_lm + ramp(step) * (aux_coef*L_balance + z_coef*L_z).  The aux
coefficient ramps from 0 -> 1x over ``aux_ramp_steps`` so the LM stabilizes
before routing pressure applies.  We log per-mode utilization, router entropy
and load-balance CV every ``eval_every`` steps to catch router collapse early.
"""

import argparse
import json
import math
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.gating.config import (
    ExperimentConfig,
    build_model_config,
    use_combined_datasets,
    use_nyx_architecture,
)
from experiments.gating.data import (
    MixedModeDataset,
    build_mixed_examples,
    collate_mixed,
    split_examples,
)
from experiments.gating.metrics import router_health
from src.models.core_model import CoreModelForCausalLM


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aux_ramp(step: int, ramp_steps: int) -> float:
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, step / ramp_steps)


def make_loaders(exp: ExperimentConfig, tokenizer, device: str):
    examples = build_mixed_examples(
        tokenizer,
        reasoning_csv=exp.reasoning_csv,
        tool_csv=exp.tool_csv,
        plain_csv=exp.plain_csv,
        max_length=exp.max_length,
        max_samples_per_source=exp.max_samples_per_source,
    )
    train, val, test = split_examples(
        examples, exp.val_fraction, exp.test_fraction, exp.seed
    )
    print(f"Examples — train {len(train)}, val {len(val)}, test {len(test)}")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collate = partial(collate_mixed, pad_id=pad_id, device=device)
    dl = lambda ds, sh: DataLoader(
        MixedModeDataset(ds), batch_size=exp.batch_size, shuffle=sh, collate_fn=collate
    )
    return dl(train, True), dl(val, False), dl(test, False)


@torch.no_grad()
def quick_eval(model, loader, max_batches: int = 20):
    model.eval()
    total_lm, total_tok = 0.0, 0
    gate_accum = None
    n_gate = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        ntok = int((batch["labels"] != -100).sum())
        total_lm += float(out["lm_loss"]) * ntok
        total_tok += ntok
        if out["gate_weights"] is not None:
            h = router_health(out["gate_weights"], batch["attention_mask"])
            gate_accum = h if gate_accum is None else {k: gate_accum[k] + h[k] for k in h}
            n_gate += 1
    model.train()
    ppl = math.exp(total_lm / max(total_tok, 1))
    health = {k: v / max(n_gate, 1) for k, v in (gate_accum or {}).items()}
    return {"val_ppl": ppl, "val_lm_loss": total_lm / max(total_tok, 1), **health}


def train(exp: ExperimentConfig):
    set_seed(exp.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(exp.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = build_model_config(exp)
    model = CoreModelForCausalLM(model_config)

    # Optionally warm-start from transferred pretrained weights. strict=False so
    # only the backbone/head load; the gate keeps its fresh initialization.
    if exp.init_weights_path:
        print(f"Loading init weights (strict=False): {exp.init_weights_path}")
        state = torch.load(exp.init_weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        result = model.load_state_dict(state, strict=False)
        non_gate_missing = [k for k in result.missing_keys if ".gates." not in k]
        print(f"  loaded. missing={len(result.missing_keys)} "
              f"(non-gate missing={len(non_gate_missing)}), unexpected={len(result.unexpected_keys)}")
        if non_gate_missing:
            print(f"  [WARN] non-gate tensors not initialized: {non_gate_missing[:8]}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M | gate layers: {model_config.gate_layer_indices}")

    train_loader, val_loader, test_loader = make_loaders(exp, tokenizer, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)
    total_steps = max(exp.num_epochs * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=exp.lr, total_steps=total_steps,
        pct_start=min(0.3, exp.warmup_steps / total_steps), anneal_strategy="cos",
    )

    os.makedirs(os.path.dirname(exp.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(exp.log_path), exist_ok=True)
    log_f = open(exp.log_path, "w", encoding="utf-8")

    step = 0
    best_ppl = float("inf")
    model.train()
    for epoch in range(exp.num_epochs):
        for batch in train_loader:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            ramp = aux_ramp(step, exp.aux_ramp_steps)
            loss = out["lm_loss"] + ramp * out["aux_loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % 20 == 0:
                print(
                    f"e{epoch} s{step} | loss {float(loss):.3f} "
                    f"lm {float(out['lm_loss']):.3f} "
                    f"bal {float(out['balance_loss']):.3f} "
                    f"z {float(out['z_loss']):.3f} ramp {ramp:.2f}"
                )

            if step > 0 and step % exp.eval_every == 0:
                metrics = quick_eval(model, val_loader)
                metrics.update({"step": step, "epoch": epoch, "train_loss": float(loss)})
                print(f"  [eval] {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})}")
                log_f.write(json.dumps(metrics) + "\n")
                log_f.flush()
                if metrics["val_ppl"] < best_ppl:
                    best_ppl = metrics["val_ppl"]
                    torch.save(
                        {"model": model.state_dict(), "config": vars(exp), "step": step},
                        exp.checkpoint_path,
                    )
                    print(f"  ✓ best val_ppl={best_ppl:.3f} — saved {exp.checkpoint_path}")
            step += 1

    log_f.close()
    print(f"Done. Best val_ppl={best_ppl:.3f}")
    return model, tokenizer, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny fast run for sanity")
    parser.add_argument("--combined", action="store_true",
                        help="train on the combined HF datasets (scripts/download_datasets.py)")
    parser.add_argument("--nyx", action="store_true",
                        help="use the nyx_reasoning architecture (1024/24/16) so pretrained weights fit")
    parser.add_argument("--init-weights", default=None,
                        help="path to transferred weights, e.g. models/nyx_gated_agentic.pth")
    args = parser.parse_args()

    exp = ExperimentConfig()
    if args.nyx:
        use_nyx_architecture(exp)
    if args.combined:
        use_combined_datasets(exp)
    if args.init_weights:
        exp.init_weights_path = args.init_weights
    if args.smoke:
        exp.num_hidden_layers = 4
        exp.hidden_size = 128
        exp.intermediate_size = 384
        exp.num_attention_heads = 4
        exp.num_key_value_heads = 4
        exp.gate_layer_indices = [2]
        exp.max_position_embeddings = 512
        exp.max_length = 512
        exp.num_epochs = 1
        exp.max_samples_per_source = 32
        exp.eval_every = 20
        exp.aux_ramp_steps = 20
        exp.warmup_steps = 5

    train(exp)


if __name__ == "__main__":
    main()
