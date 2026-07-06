# Multi-Turn Reasoning + Tool-Use Experiment — Design

**Date:** 2026-07-06
**Status:** Implemented

## Goal

Train the gated `CoreModel` on `data/tool-use-multiturn-reasoning.csv` so it can
sustain **multi-turn** interactions that interleave **reasoning**, **tool
selection**, and **response generation**, and implement the **appropriate
evaluation** for each of those capabilities.

## Dataset

ToolAce/Hermes-style ShareGPT conversations (14,579 rows). Columns:
`conversations` (list of role/value messages), `tools` (JSON function
signatures), `task`, `category`, `source`.

Roles and counts: `system` (8,026), `human` (22,852), `gpt` (51,344),
`tool` (23,837). Median 7 messages / conversation (up to 12). Assistant (`gpt`)
turns embed `<think>...</think>` reasoning and either `<tool_call>...</tool_call>`
calls or a natural-language answer; `tool` turns carry `<tool_response>` results.

**Parsing note:** the `conversations` cell is not a clean Python literal (raw
newlines used both inside values and as separators). We extract messages by the
unambiguous boundary marker `{'from': '` — robust to nested quotes and embedded
JSON. Validated on all 14,579 rows.

## Mapping to gate modes

| Assistant content | Mode |
|---|---|
| `<think>...</think>` | THINK |
| `<tool_call>...</tool_call>` | TOOL |
| answer text | RESPOND |
| end-of-turn marker | DONE |

Only assistant turns are supervised (LM loss); system/human/tool are context
(`-100`). Segment ids are eval-only — routing stays **latent** (trained with LM
loss + load-balancing aux losses, as in the base gating experiment).

## Sequence construction

A whole conversation → one token stream with a light chat template
(`<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|end|>`). The
end-of-turn marker after an assistant turn is supervised and labeled DONE so the
gate learns a per-turn stop signal. Truncate to `max_length` (default 2048),
keeping the system prompt (which holds the tool catalog).

## Model

Gated `CoreModelForCausalLM`, warm-started from `nyx_gated_agentic.pth`
(`--nyx --init-weights`), gate at mid-stack. Loss:

```
L = L_lm(assistant tokens) + ramp(step) · (aux_coef·L_balance + z_coef·L_z)
```

## Evaluation (the core requirement)

Multi-turn tool-use needs capability-specific metrics, not just perplexity:

1. **LM + per-segment perplexity** — think / tool / respond separately.
2. **Gate mode alignment** — gate argmax vs held-out segments (NMI / purity /
   confusion). Confirms the latent gate rediscovers the modes across turns.
3. **Tool-call quality** — name precision/recall/F1, argument exact-match,
   structural validity, from teacher-forced per-turn generation.
4. **Turn-decision accuracy** — per assistant turn, tool-call vs direct answer
   (2×2 confusion). This is the "tool selection vs response" decision.
5. **Tool-selection arm** — reconstructed BART-MNLI zero-shot top-1 / recall on
   gold tool turns, tying the dedicated tool arm to this dataset.

Groups 1–2 are teacher-forced over the tokenized test set; groups 3–5 use
teacher-forced per-turn generation (gold context → generate the turn → score).

## Files

`experiments/multiturn/{parse,data,config,metrics,train,evaluate}.py` +
`README.md`. Reuses `experiments/gating/metrics.py` (router health, mode
alignment) and `src/models/tool_selection.py` (BART arm).

## Validation

Smoke-tested end-to-end on the real CSV: 40 conversations produced all four
mode segments (THINK 48k / TOOL 6k / RESPOND 14k / DONE ~1k tokens); per-segment
perplexity, router health, mode alignment, tool-call extraction, and
turn-decision confusion all compute correctly.
