# Multi-Turn Reasoning + Tool-Use Experiment

Trains and evaluates the gated **CoreModel** on
[`data/tool-use-multiturn-reasoning.csv`](../../data/tool-use-multiturn-reasoning.csv)
(ToolAce/Hermes format, 14,579 conversations) so it can carry out **multi-turn**
interactions that interleave **reasoning (THINK)**, **tool calls (TOOL)**, and
**responses (RESPOND)**, ending each turn with **DONE**.

Design doc: [docs/plans/2026-07-06-multiturn-tooluse-experiment-design.md](../../docs/plans/2026-07-06-multiturn-tooluse-experiment-design.md)

## Data → modes

Each conversation is a list of `system / human / gpt / tool` messages. Only
assistant (`gpt`) turns are supervised; the assistant tags map onto the gate's
modes:

| Assistant content | Segment / mode |
|---|---|
| `<think>...</think>` | THINK |
| `<tool_call>...</tool_call>` | TOOL |
| natural-language answer | RESPOND |
| end-of-turn marker | DONE |

`system` (instructions + `<tools>`), `human`, and `tool` (`<tool_response>`)
messages are context (labels `-100`). Segment ids are used **only for
evaluation** (mode-alignment) — training routing stays latent.

## Components

| File | Role |
|---|---|
| `parse.py` | Robust conversation parser (boundary-marker extraction), tool parsing, segment tagging, tool-call extraction |
| `data.py` | `MultiTurnDataset` — chat template, assistant-only loss, per-token mode segments; `build_eval_records` for per-turn generation |
| `config.py` | `MultiTurnConfig`, `use_nyx_architecture`, `build_model_config` |
| `metrics.py` | Per-segment perplexity, tool-call scoring, turn decision, tool-selection-arm scoring |
| `train.py` | Training loop (assistant-only LM loss + ramped gate aux losses) |
| `evaluate.py` | The five evaluation groups below |

## Evaluation (five groups)

1. **LM + per-segment perplexity** — overall and separately for
   think / tool / respond, so you can see each capability improve.
2. **Gate mode alignment** — gate argmax vs held-out segment labels
   (NMI, purity, confusion matrix): did the latent gate rediscover the modes in
   a real multi-turn setting?
3. **Tool-call quality** — name precision/recall/F1, argument exact-match,
   structural validity, from teacher-forced per-turn generation.
4. **Turn-decision accuracy** — per assistant turn, did the model pick the right
   high-level action (call a tool vs answer directly)? Reported as a 2×2
   confusion.
5. **Tool-selection arm** — optional: the reconstructed BART-MNLI zero-shot
   selector's top-1 / recall on gold tool turns
   ([src/models/tool_selection.py](../../src/models/tool_selection.py)).

## Running

```bash
# Fast sanity run (tiny model, 64 conversations)
python -m experiments.multiturn.train --smoke
python -m experiments.multiturn.evaluate --smoke

# Warm-started, full nyx architecture
python -m experiments.multiturn.train --nyx --init-weights models/nyx_gated_agentic.pth
python -m experiments.multiturn.evaluate --nyx \
    --checkpoint experiments/multiturn/checkpoints/core_model_multiturn.pt \
    --tool-selector models/bart_mnli_tool_selector.pth
```

## Notes

- Groups 1–2 are teacher-forced over the tokenized test set; groups 3–4 use
  teacher-forced **per-turn generation** (gold context up to each assistant turn
  → generate that turn → score).
- Long conversations are truncated to `max_length` (default 2048); the system
  prompt carries the tool catalog and is kept.
- Watch `frac_mode_i` and `balance_cv` during training to catch router collapse.
