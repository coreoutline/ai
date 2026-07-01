"""Transfer facebook/bart-large-mnli weights into the reconstructed PyTorch model.

The reconstruction in ``src/models/bart_mnli.py`` uses the exact HF submodule
names, so this is a verified 1:1, tensor-by-tensor copy. After copying we run
BOTH the HF model and the reconstruction on the same NLI input and assert the
3-way logits match — proof the reconstruction is faithful. The reconstructed
state dict is saved to ``models/bart_mnli_tool_selector.pth``.

Usage:
    python -m scripts.transfer_bart_mnli
    python -m scripts.transfer_bart_mnli --dst models/bart_mnli_tool_selector.pth
"""

import argparse

import torch

from src.models.bart_mnli import BartForSequenceClassification, BartMnliConfig


HF_NAME = "facebook/bart-large-mnli"


@torch.no_grad()
def transfer(dst: str, atol: float = 1e-3):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading HF model: {HF_NAME}")
    hf = AutoModelForSequenceClassification.from_pretrained(HF_NAME)
    hf.eval()
    hf_sd = hf.state_dict()

    print("Building reconstructed BartForSequenceClassification ...")
    model = BartForSequenceClassification(BartMnliConfig())
    model.eval()
    tgt_sd = model.state_dict()

    # --- layer-by-layer copy with shape validation ---
    copied, shape_mismatch, missing_in_hf = [], [], []
    for key, tgt_tensor in tgt_sd.items():
        if key not in hf_sd:
            missing_in_hf.append(key)
            continue
        src_tensor = hf_sd[key]
        if src_tensor.shape != tgt_tensor.shape:
            shape_mismatch.append((key, tuple(src_tensor.shape), tuple(tgt_tensor.shape)))
            continue
        tgt_sd[key] = src_tensor.clone()
        copied.append(key)

    unexpected = [k for k in hf_sd if k not in tgt_sd]

    model.load_state_dict(tgt_sd, strict=True)

    print("\n===== Transfer report =====")
    print(f"Copied         : {len(copied)} / {len(tgt_sd)} tensors")
    print(f"Missing in HF  : {len(missing_in_hf)}")
    for k in missing_in_hf:
        print(f"    ? {k}")
    print(f"Shape mismatch : {len(shape_mismatch)}")
    for k, s, t in shape_mismatch:
        print(f"    ! {k}: hf{s} vs recon{t}")
    print(f"Unexpected(HF) : {len(unexpected)}")
    for k in unexpected[:10]:
        print(f"    - {k}")

    # --- numerical equivalence check on a real NLI pair ---
    print("\n===== Numerical verification =====")
    tok = AutoTokenizer.from_pretrained(HF_NAME)
    premise = "What is the weather in Tokyo tomorrow?"
    hypothesis = "This request requires a tool that can get weather forecasts."
    enc = tok(premise, hypothesis, return_tensors="pt", truncation=True)

    hf_logits = hf(**enc).logits
    recon_logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    print("HF logits    :", hf_logits.tolist())
    print("Recon logits :", recon_logits.tolist())
    max_diff = (hf_logits - recon_logits).abs().max().item()
    print(f"max abs diff : {max_diff:.6e}")

    ok = torch.allclose(hf_logits, recon_logits, atol=atol)
    id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
    hf_pred = id2label[int(hf_logits.argmax())]
    recon_pred = id2label[int(recon_logits.argmax())]
    print(f"HF pred={hf_pred}  recon pred={recon_pred}  allclose(atol={atol})={ok}")
    if not ok:
        raise SystemExit(f"[FAIL] logits diverge (max diff {max_diff:.3e}). Reconstruction mismatch.")
    print("[OK] Reconstruction matches HF within tolerance.")

    torch.save(model.state_dict(), dst)
    print(f"\nSaved reconstructed tool-selector weights -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst", default="models/bart_mnli_tool_selector.pth")
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()
    transfer(args.dst, atol=args.atol)


if __name__ == "__main__":
    main()
