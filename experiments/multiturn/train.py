"""Train the gated CoreModel on multi-turn reasoning + tool-use conversations.

Only assistant turns are supervised; the latent gate learns to route each token
to THINK / TOOL / RESPOND / DONE across turns. Loss:

    L = L_lm(assistant tokens) + ramp(step) * (aux_coef*L_balance + z_coef*L_z)

Usage:
    python -m experiments.multiturn.train --smoke
    python -m experiments.multiturn.train --nyx --init-weights models/nyx_gated_agentic.pth
"""

import argparse
import json
import math
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.gating.metrics import router_health
from experiments.multiturn.config import MultiTurnConfig, build_model_config, use_nyx_architecture
from experiments.multiturn.data import (
    MultiTurnDataset,
    build_examples,
    collate,
    split_examples,
)
from experiments.multiturn.metrics import perplexity_from_totals, segment_nll
from src.models.core_model import CoreModelForCausalLM


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aux_ramp(step, ramp_steps):
    return 1.0 if ramp_steps <= 0 else min(1.0, step / ramp_steps)


@torch.no_grad()
def quick_eval(model, loader, max_batches=15):
    model.eval()
    lm_nll, lm_tok = 0.0, 0
    seg_tot = {}
    gate_accum, n_gate = None, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
        ntok = int((b["labels"] != -100).sum())
        lm_nll += float(out["lm_loss"]) * ntok
        lm_tok += ntok
        sn = segment_nll(out["logits"], b["labels"], b["segment_ids"])
        for k, (nll, n) in sn.items():
            a = seg_tot.setdefault(k, [0.0, 0])
            a[0] += nll; a[1] += n
        if out["gate_weights"] is not None:
            h = router_health(out["gate_weights"], b["attention_mask"])
            gate_accum = h if gate_accum is None else {k: gate_accum[k] + h[k] for k in h}
            n_gate += 1
    model.train()
    res = {"val_ppl": math.exp(lm_nll / max(lm_tok, 1))}
    res.update(perplexity_from_totals(seg_tot))
    if n_gate:
        res.update({k: v / n_gate for k, v in gate_accum.items()})
    return res


def train(cfg: MultiTurnConfig):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("Building examples ...")
    examples = build_examples(tok, cfg.csv_path, cfg.max_length, cfg.max_samples)
    train_ex, val_ex, test_ex = split_examples(examples, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    print(f"Examples — train {len(train_ex)}, val {len(val_ex)}, test {len(test_ex)}")

    pad_id = tok.pad_token_id or tok.eos_token_id
    coll = partial(collate, pad_id=pad_id, device=device)
    dl = lambda ds, sh: DataLoader(MultiTurnDataset(ds), batch_size=cfg.batch_size, shuffle=sh, collate_fn=coll)
    train_loader, val_loader = dl(train_ex, True), dl(val_ex, False)

    model_cfg = build_model_config(cfg, tok.vocab_size, pad_id, tok.eos_token_id)
    model = CoreModelForCausalLM(model_cfg)
    if cfg.init_weights_path:
        print(f"Loading init weights (strict=False): {cfg.init_weights_path}")
        state = torch.load(cfg.init_weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        r = model.load_state_dict(state, strict=False)
        ng = [k for k in r.missing_keys if ".gates." not in k]
        print(f"  loaded; non-gate missing={len(ng)}, unexpected={len(r.unexpected_keys)}")
    model = model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | gate layers {model_cfg.gate_layer_indices}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(cfg.num_epochs * len(train_loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=0.1)

    os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
    log_f = open(cfg.log_path, "w", encoding="utf-8")

    step, best = 0, float("inf")
    model.train()
    for epoch in range(cfg.num_epochs):
        for b in train_loader:
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
            loss = out["lm_loss"] + aux_ramp(step, cfg.aux_ramp_steps) * out["aux_loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()

            if step % 20 == 0:
                print(f"e{epoch} s{step} loss {float(loss):.3f} lm {float(out['lm_loss']):.3f} "
                      f"bal {float(out['balance_loss']):.3f} z {float(out['z_loss']):.3f}")
            if step > 0 and step % cfg.eval_every == 0:
                m = quick_eval(model, val_loader)
                m.update({"step": step, "epoch": epoch})
                print("  [eval]", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}))
                log_f.write(json.dumps(m) + "\n"); log_f.flush()
                if m["val_ppl"] < best:
                    best = m["val_ppl"]
                    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step}, cfg.checkpoint_path)
                    print(f"  saved best val_ppl={best:.3f}")
            step += 1

    log_f.close()
    print(f"Done. best val_ppl={best:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--nyx", action="store_true")
    p.add_argument("--init-weights", default=None)
    args = p.parse_args()

    cfg = MultiTurnConfig()
    if args.nyx:
        use_nyx_architecture(cfg)
    if args.init_weights:
        cfg.init_weights_path = args.init_weights
    if args.smoke:
        cfg.hidden_size, cfg.intermediate_size = 128, 384
        cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads = 4, 4, 4
        cfg.gate_layer_indices = [2]
        cfg.max_position_embeddings, cfg.max_length = 1024, 1024
        cfg.num_epochs, cfg.max_samples = 1, 64
        cfg.eval_every, cfg.aux_ramp_steps = 20, 20
    train(cfg)


if __name__ == "__main__":
    main()
