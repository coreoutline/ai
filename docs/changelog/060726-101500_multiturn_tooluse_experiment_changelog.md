# Changelog — Multi-Turn Reasoning + Tool-Use Experiment

**Date:** 2026-07-06
**Feature:** Training + evaluation experiment for multi-turn generation that
interleaves reasoning, tool selection, and response, on
`data/tool-use-multiturn-reasoning.csv`, using the gated CoreModel.

## Summary

Built a complete experiment that teaches the gated `CoreModel` to handle
multi-turn tool-use conversations (ToolAce/Hermes format), with the latent gate
routing tokens across THINK / TOOL / RESPOND / DONE, and a five-group evaluation
covering every capability the task requires.

## New files

- `experiments/multiturn/parse.py` — robust conversation parser (boundary-marker
  extraction, validated on all 14,579 rows), tool parsing, segment tagging,
  tool-call extraction.
- `experiments/multiturn/data.py` — `MultiTurnDataset` (chat template,
  assistant-only loss, per-token mode segments), `build_eval_records`,
  `render_context`.
- `experiments/multiturn/config.py` — `MultiTurnConfig`, `use_nyx_architecture`,
  `build_model_config`.
- `experiments/multiturn/metrics.py` — per-segment perplexity, tool-call scoring
  (name F1 / arg exact-match / structural validity), turn-decision confusion,
  tool-selection-arm scoring.
- `experiments/multiturn/train.py` — training loop (assistant-only LM loss +
  ramped gate aux losses; `--smoke / --nyx / --init-weights`).
- `experiments/multiturn/evaluate.py` — five evaluation groups.
- `experiments/multiturn/README.md`
- `docs/plans/2026-07-06-multiturn-tooluse-experiment-design.md`

## Evaluation groups

1. LM + per-segment perplexity (think / tool / respond).
2. Gate mode alignment vs held-out segments (NMI / purity / confusion).
3. Tool-call quality (name P/R/F1, argument exact-match, structural validity).
4. Turn-decision accuracy (tool-call vs direct answer, 2×2 confusion).
5. Tool-selection arm (BART-MNLI zero-shot top-1 / recall on gold tool turns).

## Verification

Smoke-tested on the real CSV with a stub tokenizer + tiny CoreModel:
- Parser: 14,579/14,579 conversations parsed.
- 40 conversations → segment tokens THINK 48,313 / TOOL 6,066 / RESPOND 14,287 /
  DONE 968 (rest masked context).
- Per-segment perplexity, router health, mode alignment, tool-call extraction,
  and turn-decision confusion all compute correctly.
- Gold tool-call extraction verified: `{'name': 'get_future_events',
  'arguments': {'page': 1}}`.

## How to run

```bash
python -m experiments.multiturn.train --smoke
python -m experiments.multiturn.evaluate --smoke
# full, warm-started:
python -m experiments.multiturn.train --nyx --init-weights models/nyx_gated_agentic.pth
python -m experiments.multiturn.evaluate --nyx \
    --checkpoint experiments/multiturn/checkpoints/core_model_multiturn.pt \
    --tool-selector models/bart_mnli_tool_selector.pth
```

## Notes / follow-ups

- Groups 3–5 use teacher-forced per-turn generation; full free-running multi-turn
  rollout (executing tools and feeding results back) is a future extension.
- KV-cache is still not threaded through `GatedCoreModel`, so per-turn generation
  recomputes context — fine for eval, worth adding for speed.
