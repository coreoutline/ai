# Gating Mechanism + CoreModel Design

**Date:** 2026-07-01
**Author:** ML Engineering
**Status:** Approved for implementation

## 1. Motivation

The existing `CoreOutlineForCausalLM` is a standard Qwen-style decoder-only
transformer (RoPE, GQA attention, SwiGLU MLP, RMSNorm). It generates one flat
token stream and has no explicit notion of *what kind of work* it is doing at a
given step.

We want the model to internally decide, at every position, between four modes of
behavior:

| Mode      | Id | Meaning                                             |
|-----------|----|-----------------------------------------------------|
| `THINK`   | 0  | Internal reasoning / chain-of-thought               |
| `TOOL`    | 1  | Emitting a tool / function call                     |
| `RESPOND` | 2  | Producing the user-facing answer                    |
| `DONE`    | 3  | Turn is complete — a learned, mode-level stop signal|

This is a **per-position latent router**: no mode labels are used during
training. The router learns to partition its own computation using the language
modeling objective plus load-balancing auxiliary losses (MoE-style). The design
choices (confirmed with the user):

- **Granularity:** per-position router head (steers each decode step).
- **Supervision:** latent + load-balancing aux loss (no mode labels in training).
- **Model wiring:** new integrated architecture (`CoreModel`), gate woven into
  the decoder stack.
- **Conditioning:** chosen mode maps to a learned embedding added to the residual
  stream.

## 2. Gating Module (`src/core/gating.py`)

Pure, side-effect-free `nn.Module` operating on hidden states `[B, T, H]`.

**Router.** `Linear(H -> N=4)` -> per-token mode logits. During training, add
tunable Gaussian noise to logits (*noisy gating*, Shazeer et al. 2017) for
exploration, then softmax with temperature -> gate weights `g in [B, T, 4]`.

**Mode conditioning.** Learned mode table `M in [4, H]` (`nn.Parameter`). The
vector injected back into the residual is the soft mixture `mode_ctx = g @ M`.
With `use_straight_through=True`, the forward pass uses a hard one-hot
(`argmax`) while gradients flow through the soft weights (straight-through
estimator), giving near-discrete routing that is still differentiable.

**Auxiliary losses** (prevent router collapse without labels):
- **Load balance** (Switch Transformer, Fedus et al. 2021), generalized to a
  configurable per-mode target prior so `DONE` is not forced to fire ~25% of the
  time:
  `L_bal = N * sum_i (f_i * P_i) / target_i`, where `f_i` = fraction of tokens
  whose argmax is mode `i`, `P_i` = mean router prob for mode `i`. Default target
  `[1/3, 1/3, 1/3, ~1/T]` (T = expected sequence length) so `DONE` fires roughly
  once per sequence.
- **Router z-loss** (ST-MoE, Zoph et al. 2022): `L_z = mean(logsumexp(logits)^2)`
  keeps router logits from drifting to large magnitudes.

**Return:** `(mode_ctx [B,T,H], gate_weights [B,T,4], aux {"balance", "z"})`.

## 3. CoreModel (`src/models/core_model.py`)

A from-scratch integrated decoder reusing the tested primitives
(`CoreOutlineRotaryEmbedding`, `CoreOutlineAttention`, `CoreOutlineMLP`,
RMSNorm) but weaving the gate into the stack.

- `N` decoder layers as today.
- The gate is applied after the decoder layers whose index is in
  `gate_layer_indices` (default: a single gate at `N // 2`). Rationale: lower
  layers accumulate enough context to decide a mode; upper layers then *act* on
  the mode-conditioned residual before the LM head.

Per gated layer:
```
h = decoder_layer(h)          # normal attention + MLP
mode_ctx, g, aux = gate(h)    # per-position router
h = h + mode_ctx              # inject mode embedding into residual
```

By default a single shared `GatingModule` is weight-tied across gate layers
(`share_gate=True`); can be made independent.

**Outputs (dict):** `logits`, `last_hidden_state`, `gate_weights`
(`[num_gate_layers, B, T, 4]`), `aux_loss` (summed balance + z across gate
layers, each scaled by its coef), and `loss` when `labels` given:
```
loss = L_lm + router_aux_loss_coef * L_bal + router_z_loss_coef * L_z
```

**Config** (`CoreModelConfig`, extends the Qwen config fields): `num_modes=4`,
`gate_layer_indices`, `share_gate=True`, `router_aux_loss_coef=1e-2`,
`router_z_loss_coef=1e-3`, `gate_noise_std=0.3`, `gate_temperature=1.0`,
`use_straight_through=False`, `mode_balance_target`, `done_mode_id=3`.

**Generation.** `generate` stops when the top gate at the last position hits
`DONE` above `done_threshold`, falling back to EOS / `max_new_tokens`.

## 4. Experiment (`experiments/gating/`)

Routing is latent, so the experiment must (a) train stably and (b) prove the
discovered modes are meaningful using segment labels held out for eval only.

**Data (`data.py`).** `MixedModeDataset` unifies three behaviors so every mode
has reason to fire: reasoning (`thinking` -> answer), tool-calling
(`tools` -> `answers`), plain-response (answer, no thinking). Each token carries
an **eval-only** `segment_id in {think, tool, respond, done}` derived from the
existing formatters' section boundaries. These labels are stripped from the
training batch and passed only to the evaluator.

**Training (`train.py`).** Small config (`hidden=512, layers=8, heads=8`) for
feasibility. Joint loss with an **aux-loss ramp** (0 -> target over warmup steps)
so the LM stabilizes before routing pressure applies. Logs per-mode utilization,
router entropy, and load-balance CV to catch collapse early. Checkpoint + resume.

**Evaluation (`evaluate.py`).** Four metric groups:
1. **LM quality:** val/test perplexity.
2. **Router health:** per-mode utilization, mean entropy, balance CV, `DONE`
   fire-rate per sequence.
3. **Mode alignment (key eval):** held-out `segment_id` vs `argmax` gate ->
   **NMI, purity, confusion matrix**. Answers "did the latent gate rediscover
   think/tool/respond/done without supervision?"
4. **Downstream:** tool-call structural validity + exact match, reasoning-answer
   match.

## 5. Naming / integration notes

- `CoreModel` already exists (`src/models/model.py`). To avoid shadowing, the new
  classes are `GatedCoreModel` (backbone), `CoreModelForCausalLM` (LM head),
  `CoreModelConfig`. Exported additively; existing imports unchanged.

## 6. References

- Shazeer et al. 2017, *Outrageously Large Neural Networks* (noisy top-k gating).
- Fedus et al. 2021, *Switch Transformers* (load-balancing loss).
- Zoph et al. 2022, *ST-MoE* (router z-loss, stability).
