# Changelog — Gating Mechanism + CoreModel

**Date:** 2026-07-01
**Feature:** Per-position latent gating mechanism, gated `CoreModel`, and a
training/evaluation experiment.

## Summary

Added a mode-routing gate that decides, at every token position, between
`THINK`, `TOOL`, `RESPOND`, and `DONE`; wove it into a new decoder-only
`CoreModel`; and built a latent-routing experiment that verifies the gate
rediscovers the modes without supervision.

## New files

- `src/core/gating.py` — `GatingModule`, `Mode` enum (4 modes), load-balance
  (target-prior aware) + z auxiliary losses, straight-through option, noisy
  gating.
- `src/models/core_model.py` — `CoreModelConfig`, `GatedCoreModel` backbone,
  `CoreModelForCausalLM` (LM head + aux losses + DONE-aware `generate`),
  `create_core_model`.
- `experiments/gating/` — `config.py`, `data.py` (`MixedModeDataset`),
  `metrics.py`, `train.py`, `evaluate.py`, `README.md`.
- `docs/plans/2026-07-01-gating-mechanism-design.md` — approved design.

## Modified files

- `src/core/__init__.py` — export `GatingModule, Mode, NUM_MODES`.
- `src/models/__init__.py` — export `GatedCoreModel, CoreModelForCausalLM,
  CoreModelConfig, create_core_model`.

## Design decisions (confirmed with user)

- **Per-position router** (not sequence-level or soft-MoE blend).
- **Latent supervision** — no mode labels; load-balancing aux loss.
- **New integrated architecture** — gate woven into decoder stack.
- **Mode embedding into residual** — chosen mode injected into hidden state.
- **4th mode `DONE`** — learned, mode-level stop signal; halts generation and
  gets a small load-balance target so it fires ~once per sequence.

## Naming

`CoreModel` already existed (`src/models/model.py`). The gated architecture is
exported as `GatedCoreModel` / `CoreModelForCausalLM` / `CoreModelConfig` to
avoid shadowing existing imports.

## How to run

```bash
python -m experiments.gating.train --smoke
python -m experiments.gating.evaluate --smoke
```

## Follow-ups / open items

- Downstream metrics (tool-call exact match, reasoning-answer accuracy) are
  stubbed for extension in `evaluate.py`.
- KV-cache is not yet threaded through `GatedCoreModel.forward` (generation
  recomputes the prefix each step); acceptable for the experiment, worth adding
  for production inference.
