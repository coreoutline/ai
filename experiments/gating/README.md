# Gating Mechanism Experiment

Trains and evaluates **CoreModel** — a decoder-only transformer with a
**per-position latent gate** that routes each token between four behavioral
modes: `THINK`, `TOOL`, `RESPOND`, `DONE`.

Design doc: [`docs/plans/2026-07-01-gating-mechanism-design.md`](../../docs/plans/2026-07-01-gating-mechanism-design.md)

## What this tests

Routing is **latent** — no mode labels are used in training. The router learns
to partition its own computation from the language-modeling loss plus two
auxiliary losses (load-balancing + z-loss). The central question the evaluation
answers: *did the gate rediscover think / tool / respond / done on its own?*

## Components

| File | Role |
|------|------|
| `config.py`   | `ExperimentConfig` + `build_model_config` (small, runnable model) |
| `data.py`     | `MixedModeDataset` — unifies reasoning + tool-calling + plain-response data; tags each token with an **eval-only** `segment_id` |
| `metrics.py`  | Router health (utilization, entropy, balance CV) + mode alignment (NMI, purity, confusion matrix) |
| `train.py`    | Training loop with aux-loss ramp and collapse monitoring |
| `evaluate.py` | Four metric groups (see below) |

## Data sources

- `data/reasoning_smoke_test.csv` — `prompt, context, thinking, answer` → THINK + RESPOND
- `data/xlam_function_calling_smoke_test.csv` — `query, answers, tools` → TOOL
- `data/nyx-finance-instruct.csv` — `prompts, answers` → RESPOND

Each example is `prompt + completion + eos`. LM loss is computed on the
completion only (prompt masked with `-100`). Segment labels are derived from the
section boundaries and used **only** for the alignment metric.

## Warm-starting from nyx_reasoning (no training from scratch)

The gated model reuses the `CoreOutlineDecoderLayer` backbone, so pretrained
`nyx_reasoning` weights transfer directly — only the gate is new. Produce the
warm-started checkpoint once:

```bash
python -m scripts.transfer_weights \
    --src "models/nyx_reasoning (2).pth" \
    --dst "models/nyx_gated_agentic.pth"
```

This does a name+shape-matched copy (all backbone/attention/MLP/norm/lm_head
tensors load; the `model.gates.*` params keep fresh init) and writes a sidecar
`models/nyx_gated_agentic.pth.config.json`.

## Combined training datasets

`scripts/download_datasets.py` pulls open HF datasets across finance/accounting,
data-analytics (SQL), coding, reasoning, and tool-calling, and normalizes them
into the three schemas above:

```bash
python -m scripts.download_datasets --list                 # show sources
python -m scripts.download_datasets --max-per-dataset 5000  # build the mixture
```

Outputs `data/combined_{reasoning,tools,plain}.csv` + `combined_manifest.json`.

## Running

```bash
# Fast sanity run (tiny model, 32 samples/source)
python -m experiments.gating.train --smoke
python -m experiments.gating.evaluate --smoke

# Warm-started, full nyx architecture, on the combined datasets
python -m experiments.gating.train \
    --nyx --combined --init-weights models/nyx_gated_agentic.pth
python -m experiments.gating.evaluate --checkpoint experiments/gating/checkpoints/core_model.pt
```

`--nyx` sets the 1024/24/16 architecture so the pretrained weights fit;
`--combined` points at the downloaded mixture; `--init-weights` warm-starts
(strict=False — only the gate is left at init).

## Metrics

1. **LM quality** — test perplexity.
2. **Router health** — `util_mode_i`, `frac_mode_i`, `router_entropy`,
   `balance_cv` (0 = perfectly balanced), `done_fire_rate_per_seq`.
3. **Mode alignment (key)** — `nmi`, `purity`, and a confusion matrix of
   held-out segment vs predicted mode. High NMI/purity ⇒ the latent gate
   discovered semantically meaningful modes without supervision.
4. **Downstream** — extend with tool-call structural validity / reasoning-answer
   match as needed.

## Training notes

- **Aux-loss ramp** (`aux_ramp_steps`): the balance/z coefficients ramp `0 → 1×`
  so the LM stabilizes before routing pressure is applied — avoids early
  collapse onto one mode.
- **DONE balancing:** the load-balance *target prior* gives DONE a small mass
  (`~0.04`) so it fires roughly once per sequence rather than ~25% of tokens.
- Watch `balance_cv` and `frac_mode_i` in the logs: a CV trending toward 0 with
  all four modes used is healthy; one `frac` → 1.0 means collapse (raise
  `router_aux_loss_coef` or `gate_noise_std`).
