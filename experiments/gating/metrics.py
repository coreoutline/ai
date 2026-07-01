"""Metrics for the gated CoreModel: router health and latent-mode alignment.

The alignment metrics answer the central question of a *label-free* router:
did the gate rediscover think / tool / respond / done on its own? We compare the
gate's argmax against the held-out ``segment_ids`` using NMI, purity and a
confusion matrix. These labels never touch training.
"""

from typing import Dict

import torch


def router_health(gate_weights: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
    """Utilization, entropy and load-balance CV from soft gate weights.

    Args:
        gate_weights: ``[num_gate_layers, B, T, N]`` router probabilities.
        attention_mask: ``[B, T]`` 1 for real tokens.
    Returns a dict of scalar floats.
    """
    # Average over gate layers, then flatten (B, T).
    gw = gate_weights.mean(dim=0)  # [B, T, N]
    N = gw.shape[-1]
    mask = attention_mask.to(gw.dtype).unsqueeze(-1)  # [B, T, 1]
    denom = mask.sum().clamp_min(1.0)

    util = (gw * mask).sum(dim=(0, 1)) / denom  # [N] mean prob per mode
    # Fraction of tokens whose argmax is each mode.
    hard = torch.nn.functional.one_hot(gw.argmax(-1), N).to(gw.dtype)
    frac = (hard * mask).sum(dim=(0, 1)) / denom  # [N]

    # Per-token entropy of the router distribution (nats).
    ent = -(gw * (gw.clamp_min(1e-9)).log()).sum(-1, keepdim=True)  # [B, T, 1]
    mean_entropy = float((ent * mask).sum() / denom)

    # Coefficient of variation of hard utilization (0 = perfectly balanced).
    cv = float(frac.std(unbiased=False) / frac.mean().clamp_min(1e-9))

    out = {"router_entropy": mean_entropy, "balance_cv": cv}
    for i in range(N):
        out[f"util_mode_{i}"] = float(util[i])
        out[f"frac_mode_{i}"] = float(frac[i])
    return out


def _entropy(counts: torch.Tensor) -> float:
    p = counts / counts.sum().clamp_min(1.0)
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def mode_alignment(
    pred_modes: torch.Tensor,
    segment_ids: torch.Tensor,
    num_modes: int = 4,
) -> Dict:
    """NMI, purity and confusion matrix between gate argmax and true segments.

    Args:
        pred_modes: ``[M]`` predicted mode ids over valid positions.
        segment_ids: ``[M]`` ground-truth segment ids (same positions), no -100.
    """
    pred = pred_modes.long()
    true = segment_ids.long()

    K = num_modes
    conf = torch.zeros(K, K, dtype=torch.long)  # rows = true segment, cols = pred mode
    for t, p in zip(true.tolist(), pred.tolist()):
        if 0 <= t < K and 0 <= p < K:
            conf[t, p] += 1

    total = conf.sum().clamp_min(1)
    # Purity: for each predicted cluster take its majority true label.
    purity = float(conf.max(dim=0).values.sum()) / float(total)

    # Normalized mutual information.
    row = conf.sum(dim=1).float()  # true marginals
    col = conf.sum(dim=0).float()  # pred marginals
    n = float(total)
    mi = 0.0
    for i in range(K):
        for j in range(K):
            nij = float(conf[i, j])
            if nij > 0:
                mi += (nij / n) * math_log((nij * n) / (row[i] * col[j] + 1e-12))
    h_true = _entropy(row)
    h_pred = _entropy(col)
    nmi = mi / (0.5 * (h_true + h_pred) + 1e-12) if (h_true + h_pred) > 0 else 0.0

    return {
        "nmi": float(nmi),
        "purity": float(purity),
        "confusion_matrix": conf.tolist(),  # [true][pred]
    }


def math_log(x: float) -> float:
    import math

    return math.log(x)
