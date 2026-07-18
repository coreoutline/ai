# Zero-shot tool-selection: notebook bug fix + script conversion

## Context

`experiments/notebooks/ToolSelection Bart mnli.ipynb` trains the reconstructed
`facebook/bart-large-mnli` classifier (`src/models/bart_mnli.py`) that powers
the TOOL branch of the gated CoreModel (`src/models/tool_selection.py`). The
training cell crashed with:

```
RuntimeError: each element in list of batch should be of equal size
```

## Root cause

`data/tool-use-multiturn-expanded.csv` has a variable number of candidate
tools per row (1-8, confirmed via `tool_list` length distribution). The
notebook's `ToolSelectionDataset.__getitem__` returned one dataset item per
*row*, containing a Python list of tokenizer encodings (one per candidate
tool) and a matching list of binary labels. PyTorch's `default_collate`
requires every sample in a batch to have identical structure/length; since
row candidate counts differ, any batch mixing rows with different tool
counts failed to stack. Verified by reproducing the crash standalone with a
200-row sample and `shuffle=True` (deterministic once a batch mixed
different-length rows), and by confirming the fix (below) processes 671
flattened examples across mixed-count rows without error.

## Fix

Flattened the dataset so each training example is a single
`(conversation_so_far, tool_description)` NLI pair with one integer label,
instead of a per-row bundle of variable length:

- Label scheme changed from an ad hoc binary `0/1` list per row to the
  model's actual 3-way MNLI label space (`ENTAILMENT_LABEL = 2` if the tool
  was selected, `CONTRADICTION_LABEL = 0` otherwise) — the previous binary
  targets didn't match the model's `num_labels=3` output head.
- Applied directly in the notebook (cell 60: dataset class, cell 72: training
  loop, which also had a stray `:` left over from an in-progress edit that
  made it non-functional).

## New files (notebook -> script conversion)

- `src/training/tool_selection_data.py` — CSV parsing (`conversation_so_far`,
  `tool_list`, `selected_tools` JSON columns) and `load_tool_selection_data`,
  mirroring the `tool_calling_data.py` convention used by
  `fine_tune_tool_calling.py`.
- `experiments/zero_shot_tool_calling/train_zero_shot_tool_calling.py` —
  argparse CLI with the same best-val-loss checkpointing / early-stopping /
  cosine-LR pattern as `src/training/trainer.py`, adapted for classification
  (cross-entropy over 3 logits + accuracy) since the existing generic trainer
  is next-token-LM-specific and doesn't fit this model's
  `forward(input_ids, attention_mask)` signature.
- `experiments/zero_shot_tool_calling/infer_zero_shot_tool_calling.py` — thin
  CLI wrapper around the existing `ZeroShotToolSelector`
  (`src/models/tool_selection.py`); no new scoring logic, just argument
  plumbing (`--prompt`/`--prompt-file`, `--tools`/`--tools-file`,
  `--threshold`, `--top-k`).

## Verification

- Reproduced the original `RuntimeError` standalone against the real CSV +
  tokenizer, then confirmed the flattened dataset eliminates it (200-row
  sample, `batch_size=8`, `shuffle=True`).
- Ran `train_zero_shot_tool_calling.py` end-to-end (12 rows / 60 flattened
  pairs, 1 epoch, GPU auto-detected): loss decreased, checkpointing on
  best val_loss fired, final test-set eval printed (loss 0.517, acc 0.875).
- Ran `infer_zero_shot_tool_calling.py` against the existing
  `models/bart_mnli_tool_selector.pth` checkpoint with a weather-vs-doc tool
  pair: correctly ranked `get_weather` (0.957) over `create_doc` (0.179) and
  selected only `get_weather` at the default 0.5 threshold.
