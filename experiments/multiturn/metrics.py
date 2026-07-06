"""Metrics for multi-turn reasoning + tool-use.

Groups:
  1. Per-segment perplexity  — THINK / TOOL / RESPOND separately, so we can see
     whether the model learns reasoning, tool-calling, and answering each.
  2. Mode alignment          — gate argmax vs held-out segment labels (reuses the
     gating experiment's NMI / purity / confusion).
  3. Tool-call quality       — name F1, argument exact-match, structural validity.
  4. Turn-decision accuracy  — did the model pick the right action per turn
     (call a tool vs answer directly)?
  5. Tool-selection arm      — BART-MNLI zero-shot recall/top-1 on gold tool turns.
"""

import math
from typing import Any, Dict, List

import torch

# Reuse the alignment implementation from the gating experiment.
from experiments.gating.metrics import mode_alignment  # noqa: F401
from experiments.multiturn.parse import SEG_RESPOND, SEG_THINK, SEG_TOOL


# --------------------------------------------------------------------------- #
# 1. Per-segment perplexity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def segment_nll(logits: torch.Tensor, labels: torch.Tensor, segment_ids: torch.Tensor) -> Dict[int, List[float]]:
    """Sum NLL and token counts per segment id over one batch (shifted)."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_segs = segment_ids[:, 1:]
    logp = torch.log_softmax(shift_logits.float(), dim=-1)

    out: Dict[int, List[float]] = {}
    valid = shift_labels != -100
    for seg in (SEG_THINK, SEG_TOOL, SEG_RESPOND):
        mask = valid & (shift_segs == seg)
        if mask.sum() == 0:
            out[seg] = [0.0, 0]
            continue
        idx = shift_labels.clamp_min(0)
        tok_logp = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        nll = -(tok_logp[mask]).sum().item()
        out[seg] = [nll, int(mask.sum())]
    return out


def perplexity_from_totals(totals: Dict[int, List[float]]) -> Dict[str, float]:
    names = {SEG_THINK: "think", SEG_TOOL: "tool", SEG_RESPOND: "respond"}
    res = {}
    for seg, name in names.items():
        nll, n = totals.get(seg, [0.0, 0])
        res[f"ppl_{name}"] = math.exp(nll / n) if n > 0 else float("nan")
        res[f"tokens_{name}"] = n
    return res


# --------------------------------------------------------------------------- #
# 3. Tool-call quality
# --------------------------------------------------------------------------- #
def _call_name(c: Dict[str, Any]) -> str:
    return (c.get("name") or "").strip()


def _call_args(c: Dict[str, Any]) -> Dict[str, Any]:
    a = c.get("arguments", c.get("parameters", {}))
    return a if isinstance(a, dict) else {}


def score_tool_calls(pred_calls: List[Dict], gold_calls: List[Dict]) -> Dict[str, float]:
    """Name F1 + argument exact-match for one turn's predicted vs gold calls."""
    pred_names = [_call_name(c) for c in pred_calls]
    gold_names = [_call_name(c) for c in gold_calls]
    pset, gset = set(pred_names), set(gold_names)

    tp = len(pset & gset)
    precision = tp / len(pset) if pset else 0.0
    recall = tp / len(gset) if gset else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Argument exact-match over correctly-named calls.
    arg_matches, arg_total = 0, 0
    gold_by_name = {_call_name(c): _call_args(c) for c in gold_calls}
    for c in pred_calls:
        name = _call_name(c)
        if name in gold_by_name:
            arg_total += 1
            if _call_args(c) == gold_by_name[name]:
                arg_matches += 1
    arg_em = arg_matches / arg_total if arg_total else 0.0

    return {"name_precision": precision, "name_recall": recall, "name_f1": f1, "arg_exact_match": arg_em}


def structural_validity(pred_calls_raw: List[Any]) -> float:
    """Fraction of extracted tool-call bodies that parsed into name+arguments."""
    if not pred_calls_raw:
        return float("nan")
    valid = sum(1 for c in pred_calls_raw if isinstance(c, dict) and c.get("name") is not None)
    return valid / len(pred_calls_raw)


# --------------------------------------------------------------------------- #
# 4. Turn-decision (tool vs respond)
# --------------------------------------------------------------------------- #
def turn_action(calls: List[Dict]) -> str:
    return "tool" if calls else "respond"


def decision_confusion(pairs: List[tuple]) -> Dict[str, Any]:
    """pairs = [(gold_action, pred_action)]. Returns accuracy + 2x2 confusion."""
    labels = ["tool", "respond"]
    idx = {l: i for i, l in enumerate(labels)}
    conf = [[0, 0], [0, 0]]
    correct = 0
    for g, p in pairs:
        conf[idx[g]][idx[p]] += 1
        correct += int(g == p)
    acc = correct / len(pairs) if pairs else float("nan")
    return {"turn_decision_accuracy": acc, "confusion": conf, "labels": labels}


# --------------------------------------------------------------------------- #
# 5. Tool-selection arm (BART-MNLI zero-shot) on gold tool turns
# --------------------------------------------------------------------------- #
def tool_selection_scores(selected_names: List[str], gold_names: List[str], k: int = 1) -> Dict[str, float]:
    if not gold_names:
        return {}
    sset, gset = set(selected_names), set(gold_names)
    top1 = 1.0 if selected_names[:1] and selected_names[0] in gset else 0.0
    recall = len(sset & gset) / len(gset)
    return {"tool_sel_top1": top1, "tool_sel_recall": recall}
