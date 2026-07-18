# Combine tool-calling datasets from Hugging Face into one CSV

## What changed

Added `scripts/build_tool_calling_dataset.py`. It downloads three tool-calling
datasets from the Hugging Face Hub, normalizes each to the same
`query / tools / answers` schema already used by
[`src/training/tool_calling_data.py`](../../src/training/tool_calling_data.py)
and [`experiments/gating/data.py`](../../experiments/gating/data.py), dedups on
query text, shuffles, and writes one combined CSV plus a manifest.

Sources (see docstring for full rationale):

| name                | HF dataset                                    | raw schema                              |
|---------------------|------------------------------------------------|------------------------------------------|
| `xlam`               | `Salesforce/xlam-function-calling-60k` (gated) | already `query/tools/answers`             |
| `glaive_v2`          | `glaiveai/glaive-function-calling-v2`          | `system`/`chat` free text                 |
| `hermes_singleturn`  | `NousResearch/hermes-function-calling-v1` (config `func_calling_singleturn`) | ShareGPT-style `conversations` + `tools` |

## Why a new script instead of extending `scripts/download_datasets.py`

That script builds the *mixed-mode* training corpus for the gating experiment
(reasoning / tools / plain buckets across many domains) and already has a
`xlam_tools` source feeding its `tools` bucket. This script is narrower and
standalone: it only pulls tool-calling data, from multiple *distinct* sources
(not just xLAM), and its output CSV is usable on its own with the existing
`EncodedToolCallingDataset` / `load_tool_calling_data` loader — verified by
loading a sample row through `format_tool_calling_example`.

## Key implementation detail

Glaive-v2 and Hermes embed JSON *inside* free text
(`<functioncall> {...}`, `<tool_call>{...}</tool_call>`), and Glaive's
`arguments` field is itself a JSON string nested inside the outer object
(`{"name": "x", "arguments": '{"a": 1}'}`). A naive regex like `\{.*?\}` stops
at the *first* `}`, which truncates the outer object. Wrote
`_find_balanced_json_object`, a small brace-depth scanner that also tracks
quote state (so braces inside quoted strings don't count), and reused it for
both Glaive's function list and both datasets' embedded tool-call objects.

## Verification

- Unit-tested all three mappers against synthetic rows shaped like the real
  data (including the nested-quote edge case above) — all parsed correctly,
  and rows with no function call correctly returned `None`.
- Live smoke test (`--only hermes_singleturn glaive_v2 --max-per-dataset 20`)
  against the real HF datasets: 38 rows collected and written, columns and
  content inspected manually, then round-tripped through
  `format_tool_calling_example` to confirm the output CSV is a drop-in for
  the existing training pipeline.
- `xlam` was not exercised live: the `HUGGINGFACE_API_KEY` in `.env` is
  expired/invalid ("Invalid user token"), and `xlam-function-calling-60k` is
  a gated dataset. The script logs in via `huggingface_hub.login()` using
  that token if present and warns (non-fatally) on failure — a valid token
  is needed to pull `xlam` until this is refreshed.

## Follow-up needed

Refresh `HUGGINGFACE_API_KEY` in `.env` to pull the `xlam` source (accept the
dataset's terms on the Hub with that account first, since it's gated).
