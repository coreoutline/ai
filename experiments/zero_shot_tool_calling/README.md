# Zero-Shot Tool Selection Experiment

Trains and runs the reconstructed `facebook/bart-large-mnli` classifier that
powers the **TOOL** branch of the gated CoreModel. The user prompt is treated
as an NLI *premise*; each candidate tool's description is turned into a
*hypothesis* ("This request requires a tool that can {description}."). The
model scores every (premise, hypothesis) pair, and the entailment probability
is the tool's relevance score — the same zero-shot-classification trick the
HF `bart-large-mnli` pipeline uses, just with a reconstructed model
(`src/models/bart_mnli.py`) so it can be fine-tuned.

Converted from `experiments/notebooks/ToolSelection Bart mnli.ipynb`. See
[`docs/changelog/180726-095132_zero_shot_tool_calling_training_script_changelog.md`](../../docs/changelog/180726-095132_zero_shot_tool_calling_training_script_changelog.md)
for the collate-error root cause this conversion fixed.

## Components

| File | Role |
|------|------|
| `train_zero_shot_tool_calling.py` | Training CLI — fine-tunes the classifier with best-val-loss checkpointing, early stopping, cosine LR |
| `infer_zero_shot_tool_calling.py` | Inference CLI — scores/selects tools for a prompt using a trained checkpoint |
| `src/training/tool_selection_data.py` | Data loading (used by the training script) |
| `src/models/bart_mnli.py` | Reconstructed BART-large-MNLI architecture (3-way NLI head: contradiction/neutral/entailment) |
| `src/models/tool_selection.py` | `ZeroShotToolSelector` — the scoring/selection logic the inference script wraps, plus `run_tool_arm` for CoreModel gate integration |

## Data

`data/tool-use-multiturn-expanded.csv`. Each row is a conversation with a
variable number of candidate tools (`tool_list`, 1–8 per row) and the tool(s)
actually called (`selected_tools`). Training flattens each row into one
`(conversation_so_far, tool_description)` NLI pair per candidate tool, labeled
`entailment` if that tool was selected, `contradiction` otherwise — a per-row
item of variable length can't be batched by PyTorch's default collate, which
is what the original notebook got wrong.

## Training

```bash
python3.11 experiments/zero_shot_tool_calling/train_zero_shot_tool_calling.py     --csv-path ./data/tool-use-multiturn-expanded.csv     --checkpoint ./models/bart_mnli_tool_selector.pth     --epochs 30   --batch-size 4
```

Resumes from `--checkpoint` if it already exists, otherwise trains
`bart-large-mnli`'s architecture from random init. Key flags:

- `--max-samples N` — limit rows (before flattening) for a smoke test
- `--eval-freq` / `--eval-iter` — validation cadence and batch count per eval
- `--patience` — early-stopping steps without val-loss improvement (`0`/`None` disables)
- `--lr`, `--weight-decay`, `--scheduler-t-max`, `--scheduler-eta-min` — optimizer/LR-schedule knobs

Runs on CUDA automatically if available (BART-large is slow on CPU — expect
minutes per epoch even on small slices).

## Inference

```bash
python3.11 experiments/zero_shot_tool_calling/infer_zero_shot_tool_calling.py \
    --checkpoint ./models/bart_mnli_tool_selector.pth \
    --prompt "What is the weather in Tokyo tomorrow?" \
    --tools '[{"name": "get_weather", "description": "Get the weather forecast for a location."},
              {"name": "create_doc", "description": "Create a new Google Doc."}]' \
    --threshold 0.5 \
    --top-k 3
```

Prints every candidate tool ranked by entailment probability, the tools that
clear `--threshold` (capped by `--top-k`), and the rendered tool-signature
block that would be injected into the CoreModel's context. Tools can also
come from a file via `--tools-file path/to/tools.json` instead of `--tools`;
accepts either `{name, description}` or xLAM-style
`{"function": {name, description}}` entries. Use `--prompt-file` to read a
long prompt from disk instead of `--prompt`.
