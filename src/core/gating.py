"""Per-position latent gating mechanism.

The gate decides, at every token position, between four modes of behavior:

    THINK   (0) - internal reasoning / chain-of-thought
    TOOL    (1) - emitting a tool / function call
    RESPOND (2) - producing the user-facing answer
    DONE    (3) - the turn is complete (a learned, mode-level stop signal)

Routing is *latent*: no mode labels are used during training. The router learns
to partition its own computation from the language-modeling objective plus two
auxiliary losses that keep it from collapsing onto a single mode:

  * load-balancing loss  (Switch Transformer, Fedus et al. 2021), generalized to
    a configurable per-mode target prior so DONE is not forced to fire ~25% of
    the time;
  * router z-loss        (ST-MoE, Zoph et al. 2022) for logit stability.

The chosen mode is injected back into the residual stream as a learned mode
embedding (a soft mixture, optionally hardened with a straight-through
estimator). The module is pure and side-effect free so it can be unit tested in
isolation.
"""

from enum import IntEnum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mode(IntEnum):
    """Behavioral modes the gate routes between."""

    THINK = 0
    TOOL = 1
    RESPOND = 2
    DONE = 3


NUM_MODES = len(Mode)


class GatingModule(nn.Module):
    """Per-position router over behavioral modes with residual mode conditioning.

    Args:
        hidden_size: model hidden dimension ``H``.
        num_modes: number of behavioral modes ``N`` (default 4).
        noise_std: std of Gaussian noise added to router logits during training
            (noisy gating, Shazeer et al. 2017). Set 0 to disable.
        temperature: softmax temperature for the router.
        use_straight_through: if True the forward pass injects the hard argmax
            mode embedding while gradients flow through the soft weights.
        balance_target: optional per-mode target utilization prior of shape
            ``[num_modes]``. Defaults to uniform over the first three modes with a
            small mass on DONE (``~1/expected_seq_len``). Used to scale the
            load-balancing loss so rare modes are not over-encouraged.
        done_mode_id: index of the DONE mode (used only to build the default
            balance target).
        expected_seq_len: expected sequence length, used to set the default DONE
            target (~1 fire per sequence).
    """

    def __init__(
        self,
        hidden_size: int,
        num_modes: int = NUM_MODES,
        noise_std: float = 0.3,
        temperature: float = 1.0,
        use_straight_through: bool = False,
        balance_target: Optional[torch.Tensor] = None,
        done_mode_id: int = int(Mode.DONE),
        expected_seq_len: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modes = num_modes
        self.noise_std = noise_std
        self.temperature = temperature
        self.use_straight_through = use_straight_through
        self.done_mode_id = done_mode_id

        # Router: hidden state -> per-mode logits.
        self.router = nn.Linear(hidden_size, num_modes, bias=False)
        # Learned mode embedding table M in [num_modes, hidden_size].
        self.mode_embeddings = nn.Parameter(torch.zeros(num_modes, hidden_size))

        # Per-mode load-balance target prior.
        if balance_target is None:
            balance_target = torch.full((num_modes,), 1.0 / num_modes)
            if 0 <= done_mode_id < num_modes:
                done_share = min(1.0 / max(expected_seq_len, 1), 1.0 / num_modes)
                remaining = (1.0 - done_share) / (num_modes - 1)
                balance_target.fill_(remaining)
                balance_target[done_mode_id] = done_share
        else:
            balance_target = balance_target.clone().float()
        balance_target = balance_target / balance_target.sum()
        # Buffer so it moves with .to(device) but is not a trained parameter.
        self.register_buffer("balance_target", balance_target, persistent=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mode_embeddings, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Route each position and return the residual mode conditioning.

        Args:
            hidden_states: ``[B, T, H]`` hidden states.
            attention_mask: optional ``[B, T]`` mask of real (1) vs pad (0)
                positions. Padded positions are excluded from the auxiliary
                loss statistics.

        Returns:
            mode_ctx: ``[B, T, H]`` mode embedding to add to the residual stream.
            gate_weights: ``[B, T, num_modes]`` soft router probabilities.
            aux: dict with scalar tensors ``{"balance", "z"}``.
        """
        logits = self.router(hidden_states)  # [B, T, N]

        # Noisy gating for exploration during training only.
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # Router z-loss on the raw (pre-temperature) logits.
        z_per_token = torch.logsumexp(logits, dim=-1) ** 2  # [B, T]

        gate_weights = F.softmax(logits / self.temperature, dim=-1)  # [B, T, N]

        # Mode conditioning: soft mixture of mode embeddings.
        soft_ctx = gate_weights @ self.mode_embeddings  # [B, T, H]
        if self.use_straight_through:
            hard_idx = gate_weights.argmax(dim=-1)  # [B, T]
            hard_ctx = self.mode_embeddings[hard_idx]  # [B, T, H]
            # Straight-through: forward uses hard, backward uses soft.
            mode_ctx = hard_ctx + (soft_ctx - soft_ctx.detach())
        else:
            mode_ctx = soft_ctx

        aux = self._aux_losses(gate_weights, z_per_token, attention_mask)
        return mode_ctx, gate_weights, aux

    def _aux_losses(
        self,
        gate_weights: torch.Tensor,
        z_per_token: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Load-balancing and z auxiliary losses over unmasked positions."""
        B, T, N = gate_weights.shape

        if attention_mask is not None:
            mask = attention_mask.to(gate_weights.dtype)  # [B, T]
            mask_flat = mask.reshape(-1)  # [B*T]
            denom = mask_flat.sum().clamp_min(1.0)
        else:
            mask_flat = torch.ones(B * T, dtype=gate_weights.dtype, device=gate_weights.device)
            denom = mask_flat.sum().clamp_min(1.0)

        gw = gate_weights.reshape(-1, N)  # [B*T, N]

        # P_i: mean router probability per mode.
        P = (gw * mask_flat.unsqueeze(-1)).sum(dim=0) / denom  # [N]

        # f_i: fraction of tokens whose argmax is mode i (hard dispatch).
        hard = F.one_hot(gw.argmax(dim=-1), num_classes=N).to(gate_weights.dtype)  # [B*T, N]
        f = (hard * mask_flat.unsqueeze(-1)).sum(dim=0) / denom  # [N]

        # Target-scaled load balance: N * sum_i f_i * P_i / target_i.
        target = self.balance_target.to(gate_weights.dtype).clamp_min(1e-6)
        balance = N * torch.sum(f * P / target)

        z = (z_per_token.reshape(-1) * mask_flat).sum() / denom

        return {"balance": balance, "z": z}

    @torch.no_grad()
    def route(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Hard mode assignment ``[B, T]`` for analysis / inference control."""
        logits = self.router(hidden_states)
        return logits.argmax(dim=-1)
