# Changelog — Zero-Shot Tool-Selection Arm (reconstructed BART-large-MNLI)

**Date:** 2026-07-01
**Feature:** The TOOL branch of the gated CoreModel — a zero-shot tool selector
built on a from-scratch PyTorch reconstruction of `facebook/bart-large-mnli`.

## Summary

When the gate routes a position to `TOOL`, the model must decide *which* tools
to call. This is done exactly like `bart-large-mnli` zero-shot classification:
the prompt is the NLI premise, each candidate tool is a hypothesis, and the
entailment probability is the tool's relevance score (multi-label sigmoid). The
BART-MNLI model was **reconstructed from scratch in PyTorch** and its weights
transferred layer-by-layer from the HF checkpoint (verified numerically).

## New files

- `src/models/bart_mnli.py` — faithful PyTorch BART-large-MNLI: encoder/decoder,
  `BartAttention` (self + cross), learned positions (offset 2), tied embeddings,
  post-norm, gelu, EOS-pooled `BartClassificationHead` (3 NLI labels). Module
  names match HF exactly.
- `scripts/transfer_bart_mnli.py` — loads HF `bart-large-mnli`, copies all 517
  tensors 1:1 into the reconstruction, then asserts the 3-way logits match HF on
  a real NLI pair (**max abs diff 9.5e-07**), and saves
  `models/bart_mnli_tool_selector.pth`.
- `src/models/tool_selection.py` — `ZeroShotToolSelector` (score / select /
  render_constraint / build_tool_prompt), `load_tool_selector`, and CoreModel
  integration: `detect_tool_gate` + `run_tool_arm`.

## Modified files

- `src/models/__init__.py` — export the BART and tool-selection symbols.

## Design decisions (confirmed with user)

- **Backend:** plug-in bart-large-mnli (reconstructed), not a native head — works
  zero-shot immediately.
- **Selection:** multi-tool, sigmoid (`softmax([contradiction, entailment])`),
  threshold-based; falls back to top-1 if none clear the threshold.
- **Integration:** score **and** constrain generation — selected tool
  signatures are injected into the context before the CoreModel emits the
  function-call arguments.

## Verification

- Weight transfer: 517/517 tensors, logits match HF to < 1e-6.
- Scoring: on sample prompts the correct tool ranks top at 0.98–0.99
  (get_weather / send_email / calculate_mortgage).
- Integration: `run_tool_arm` produces gate mode, ranked scores, selected tools,
  the constrained prompt, and generated call text end-to-end.

## Usage

```bash
# One-time: reconstruct + transfer BART-MNLI weights
python -m scripts.transfer_bart_mnli   # -> models/bart_mnli_tool_selector.pth
```
```python
from src.models.tool_selection import load_tool_selector, run_tool_arm
selector = load_tool_selector("models/bart_mnli_tool_selector.pth")
result = run_tool_arm(core_model, core_tokenizer, prompt, tools, selector)
# result["selected"], result["constrained_prompt"], result["generation"]
```

## Notes / follow-ups

- Zero-shot NLI can give moderate scores to loosely-related tools; tune
  `hypothesis_template` and `threshold` per tool catalog. Top-1 ranking is
  reliable.
- `models/bart_mnli_tool_selector.pth` is ~1.6GB (fp32). Consider fp16 for
  deployment.
- Optional future work: distill BART-MNLI scores into a native CoreModel head to
  drop the separate 400M-param model at inference.
