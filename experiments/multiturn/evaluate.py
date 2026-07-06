"""Evaluate the gated CoreModel on multi-turn reasoning + tool use.

Metric groups (see metrics.py):
  1. LM + per-segment perplexity (think / tool / respond)
  2. Gate mode alignment vs held-out segments (NMI / purity / confusion)
  3. Tool-call quality (name F1, argument exact-match, structural validity)
  4. Turn-decision accuracy (call a tool vs answer directly)
  5. Tool-selection arm (BART-MNLI zero-shot) on gold tool turns

Groups 1-2 are teacher-forced over the tokenized test set. Groups 3-4 use
teacher-forced *per-turn generation*: feed gold context up to each assistant
turn, generate that turn, and score against the reference. Group 5 (optional)
runs the reconstructed BART-MNLI selector.

Usage:
    python -m experiments.multiturn.evaluate --smoke
    python -m experiments.multiturn.evaluate --checkpoint experiments/multiturn/checkpoints/core_model_multiturn.pt \
        --tool-selector models/bart_mnli_tool_selector.pth
"""

import argparse
import json
import math
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.multiturn.config import MultiTurnConfig, build_model_config, use_nyx_architecture
from experiments.multiturn.data import (
    MultiTurnDataset,
    build_eval_records,
    build_examples,
    collate,
    render_context,
    split_examples,
)
from experiments.multiturn.metrics import (
    decision_confusion,
    mode_alignment,
    perplexity_from_totals,
    score_tool_calls,
    segment_nll,
    structural_validity,
    tool_selection_scores,
    turn_action,
)
from experiments.multiturn.parse import SEGMENT_NAMES, extract_tool_calls, last_user_query
from experiments.gating.metrics import router_health
from src.models.core_model import CoreModelForCausalLM


@torch.no_grad()
def teacher_forced_metrics(model, loader):
    """LM/segment perplexity, router health, and gate mode-alignment."""
    model.eval()
    lm_nll, lm_tok = 0.0, 0
    seg_tot = {}
    gate_accum, n_gate = None, 0
    all_pred, all_seg = [], []

    for b in loader:
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
        ntok = int((b["labels"] != -100).sum())
        lm_nll += float(out["lm_loss"]) * ntok
        lm_tok += ntok
        for k, (nll, n) in segment_nll(out["logits"], b["labels"], b["segment_ids"]).items():
            a = seg_tot.setdefault(k, [0.0, 0]); a[0] += nll; a[1] += n
        gw = out["gate_weights"]
        if gw is not None:
            h = router_health(gw, b["attention_mask"])
            gate_accum = h if gate_accum is None else {k: gate_accum[k] + h[k] for k in h}
            n_gate += 1
            pred = gw.mean(dim=0).argmax(dim=-1)
            valid = b["segment_ids"] != -100
            all_pred.append(pred[valid]); all_seg.append(b["segment_ids"][valid])

    res = {"test_ppl": math.exp(lm_nll / max(lm_tok, 1))}
    res.update(perplexity_from_totals(seg_tot))
    if n_gate:
        res.update({k: v / n_gate for k, v in gate_accum.items()})
    if all_pred:
        res["mode_alignment"] = mode_alignment(torch.cat(all_pred), torch.cat(all_seg), num_modes=4)
    return res


@torch.no_grad()
def generation_metrics(model, tok, records, max_records=20, max_new_tokens=96, selector=None):
    """Per-turn teacher-forced generation: tool-call quality + turn decision."""
    model.eval()
    device = next(model.parameters()).device
    agg = {"name_f1": [], "arg_exact_match": [], "name_precision": [], "name_recall": []}
    struct = []
    decision_pairs = []
    sel_top1, sel_recall = [], []
    n_turns = 0

    for rec in records[:max_records]:
        messages, tools = rec["messages"], rec["tools"]
        for t, m in enumerate(messages):
            if m["from"] != "gpt":
                continue
            gold_calls = extract_tool_calls(m["value"])
            gold_action = turn_action(gold_calls)

            context = render_context(messages, t)
            enc = tok(context, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings)
            input_ids = enc["input_ids"].to(device)
            gen = model.generate(input_ids, max_new_tokens=max_new_tokens, eos_id=tok.eos_token_id)
            text = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)

            pred_calls = extract_tool_calls(text)
            decision_pairs.append((gold_action, turn_action(pred_calls)))
            struct.append(structural_validity(pred_calls))
            if gold_calls:
                sc = score_tool_calls(pred_calls, gold_calls)
                for k in agg:
                    agg[k].append(sc[k])

                # Tool-selection arm on this gold tool turn.
                if selector is not None and tools:
                    query = last_user_query(messages, t) or context[-400:]
                    selected = selector.select(query, tools, threshold=0.5)
                    sel_names = [s.name for s in selected]
                    gold_names = [c.get("name", "") for c in gold_calls]
                    ss = tool_selection_scores(sel_names, gold_names)
                    if ss:
                        sel_top1.append(ss["tool_sel_top1"]); sel_recall.append(ss["tool_sel_recall"])
            n_turns += 1

    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    out = {
        "gen_turns_evaluated": n_turns,
        "tool_name_f1": mean(agg["name_f1"]),
        "tool_name_precision": mean(agg["name_precision"]),
        "tool_name_recall": mean(agg["name_recall"]),
        "tool_arg_exact_match": mean(agg["arg_exact_match"]),
        "tool_struct_validity": mean([s for s in struct if not math.isnan(s)]),
    }
    out.update(decision_confusion(decision_pairs))
    if sel_top1:
        out["bart_tool_sel_top1"] = mean(sel_top1)
        out["bart_tool_sel_recall"] = mean(sel_recall)
    return out


def pretty_confusion(conf, names):
    lines = ["true\\pred  " + "  ".join(f"{n:>8}" for n in names)]
    for i, row in enumerate(conf):
        lines.append(f"{names[i]:>9}  " + "  ".join(f"{c:>8}" for c in row))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="experiments/multiturn/checkpoints/core_model_multiturn.pt")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--nyx", action="store_true")
    p.add_argument("--tool-selector", default=None, help="path to bart_mnli_tool_selector.pth")
    p.add_argument("--max-gen-records", type=int, default=20)
    p.add_argument("--no-generate", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MultiTurnConfig()
    if args.nyx:
        use_nyx_architecture(cfg)
    if args.smoke:
        cfg.hidden_size, cfg.intermediate_size = 128, 384
        cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads = 4, 4, 4
        cfg.gate_layer_indices = [2]
        cfg.max_position_embeddings, cfg.max_length = 1024, 1024
        cfg.max_samples = 64

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id or tok.eos_token_id

    model_cfg = build_model_config(cfg, tok.vocab_size, pad_id, tok.eos_token_id)
    model = CoreModelForCausalLM(model_cfg).to(device)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded checkpoint {args.checkpoint} (step {ckpt.get('step')})")
    except FileNotFoundError:
        print(f"[WARN] no checkpoint at {args.checkpoint}; evaluating randomly-initialized model.")

    examples = build_examples(tok, cfg.csv_path, cfg.max_length, cfg.max_samples)
    _, _, test_ex = split_examples(examples, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    test_loader = DataLoader(MultiTurnDataset(test_ex), batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=partial(collate, pad_id=pad_id, device=device))

    print("\n== Teacher-forced perplexity + gate alignment ==")
    tf = teacher_forced_metrics(model, test_loader)
    align = tf.pop("mode_alignment", None)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in tf.items()}, indent=2))
    if align:
        print(f"\nMode alignment: NMI={align['nmi']:.4f} purity={align['purity']:.4f}")
        print(pretty_confusion(align["confusion_matrix"], SEGMENT_NAMES))

    if not args.no_generate:
        selector = None
        if args.tool_selector:
            from src.models.tool_selection import load_tool_selector
            selector = load_tool_selector(args.tool_selector, device=device)
        print("\n== Per-turn generation: tool-call quality + turn decision ==")
        gm = generation_metrics(model, tok, build_eval_records(cfg.csv_path, cfg.max_samples),
                                max_records=args.max_gen_records, selector=selector)
        conf = gm.pop("confusion", None); labels = gm.pop("labels", None)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in gm.items()}, indent=2))
        if conf:
            print("\nTurn decision confusion:")
            print(pretty_confusion(conf, labels))


if __name__ == "__main__":
    main()
