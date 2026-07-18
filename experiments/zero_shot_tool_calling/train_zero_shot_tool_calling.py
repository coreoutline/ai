"""
Training script for the zero-shot tool-selection classifier (reconstructed
BART-large-MNLI). This is the TOOL branch of the gated CoreModel: given a
conversation-so-far as the NLI premise and a candidate tool's description as
the hypothesis, the model predicts entailment (tool is relevant) vs
contradiction (tool is not relevant).

Converted from experiments/notebooks/ToolSelection Bart mnli.ipynb, fixing the
notebook's collate bug: rows have a variable number of candidate tools (1-8),
so training must flatten to one (premise, tool_description) pair per example
rather than batching variable-length per-row lists directly.

Dataset: data/tool-use-multiturn-expanded.csv

Example:
    python3.11 experiments/zero_shot_tool_calling/train_zero_shot_tool_calling.py \\
        --csv-path ./data/tool-use-multiturn-expanded.csv \\
        --checkpoint ./models/bart_mnli_tool_selector.pth \\
        --epochs 3 \\
        --batch-size 16 \\
        --max-samples 2000
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.bart_mnli import BartForSequenceClassification, BartMnliConfig
from src.training.tool_selection_data import DEFAULT_TOKENIZER, load_tool_selection_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the zero-shot tool-selection BART-MNLI classifier"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="./data/tool-use-multiturn-expanded.csv",
        help="Path to the tool-use-multiturn-expanded CSV file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./models/bart_mnli_tool_selector.pth",
        help="Path to load/save model checkpoint",
    )
    parser.add_argument("--tokenizer-name", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--eval-iter", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset size (rows, before flattening) for smoke tests",
    )
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--scheduler-t-max", type=int, default=50)
    parser.add_argument("--scheduler-eta-min", type=float, default=6e-6)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping: eval steps without val_loss improvement before stopping (default: 5)",
    )
    return parser.parse_args()


def calc_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, num_batches):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for i, (input_batch, target_batch) in enumerate(loader):
        if i >= num_batches:
            break
        logits = model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
        )
        total_loss += F.cross_entropy(logits, target_batch).item()
        total_acc += calc_accuracy(logits, target_batch)
        n += 1
    model.train()
    if n == 0:
        return float("nan"), float("nan")
    return total_loss / n, total_acc / n


def train(model, train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.scheduler_t_max, args.epochs), eta_min=args.scheduler_eta_min
    )

    best_val_loss = float("inf")
    evals_without_improvement = 0
    global_step = -1

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1} training start...")
        model.train()
        for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            logits = model(
                input_ids=input_batch["input_ids"],
                attention_mask=input_batch["attention_mask"],
            )
            loss = F.cross_entropy(logits, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1

            if global_step % args.eval_freq == 0:
                train_loss, train_acc = evaluate(model, train_loader, args.eval_iter)
                val_loss, val_acc = evaluate(model, val_loader, args.eval_iter)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f} acc {train_acc:.3f}, "
                    f"Val loss {val_loss:.3f} acc {val_acc:.3f}, "
                    f"LR {scheduler.get_last_lr()[0]:.2e}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    evals_without_improvement = 0
                    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
                    torch.save(model.state_dict(), args.checkpoint)
                    print(f"  New best val_loss={val_loss:.3f} -- checkpoint saved to {args.checkpoint}")
                else:
                    evals_without_improvement += 1
                    print(
                        f"  No improvement ({evals_without_improvement}/"
                        f"{args.patience if args.patience else 'inf'} patience steps used)."
                    )

                if args.patience is not None and evals_without_improvement >= args.patience:
                    print(
                        f"\nEarly stopping triggered after {args.patience} eval steps "
                        f"without improvement. Best val_loss={best_val_loss:.3f}"
                    )
                    return

        scheduler.step()


def main():
    args = parse_args()
    torch.manual_seed(123)

    model = BartForSequenceClassification(BartMnliConfig())
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, tokenizer, val_data = load_tool_selection_data(
        csv_path=args.csv_path,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        device=device,
        max_samples=args.max_samples,
    )

    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        print("No checkpoint found, training from scratch (bart-large-mnli architecture, random init).")

    start_time = time.time()
    train(model, train_loader, val_loader, args)
    execution_time_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    test_loss, test_acc = evaluate(model, test_loader, len(test_loader))
    print(f"Test loss {test_loss:.3f} acc {test_acc:.3f}")


if __name__ == "__main__":
    main()
