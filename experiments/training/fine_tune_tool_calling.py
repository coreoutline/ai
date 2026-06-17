"""
Fine-tuning script for CoreOutline Qwen model on xLAM function-calling data.

Dataset: data/xlam_function_calling_60k.csv
Columns: query, answers (JSON function calls), tools (JSON tool schemas)

Example:
    python3.11 experiments/training/fine_tune_tool_calling.py \\
        --csv-path ./data/xlam_function_calling_60k.csv \\
        --checkpoint ./models/nyx_2_tool_calling.pth \\
        --epochs 3 \\
        --batch-size 4 \\
        --max-samples 1000
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core import CoreOutlineConfig, create_coreoutline_qwen_model
from src.training.tool_calling_data import format_tool_calling_prompt, load_tool_calling_data
from src.training.trainer import train_model_simple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune CoreOutline on xLAM function-calling data"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="./data/xlam_function_calling_60k.csv",
        help="Path to the xLAM CSV file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./models/nyx_2_tool_calling.pth",
        help="Path to load/save model checkpoint",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="./models/nyx_2.pth",
        help="Optional base weights to initialize from before fine-tuning",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--eval-iter", type=int, default=5)
    parser.add_argument("--allowed-max-length", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset size for smoke tests (default: use full dataset)",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping: eval steps without val_loss improvement before stopping (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(123)

    config = CoreOutlineConfig(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )
    model = create_coreoutline_qwen_model(config)

    with torch.no_grad():
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, tokenizer, val_data = load_tool_calling_data(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allowed_max_length=args.allowed_max_length,
        device=str(device),
        max_samples=args.max_samples,
    )

    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    elif os.path.exists(args.base_checkpoint):
        print(f"Loading base weights from {args.base_checkpoint}...")
        model.load_state_dict(torch.load(args.base_checkpoint, map_location="cpu"))
    else:
        print("No checkpoint found, training from scratch.")

    model.config.use_cache = False
    model.config.gradient_checkpointing = True

    start_context_idx = 10 if len(val_data) > 10 else 0
    start_context = format_tool_calling_prompt(val_data.iloc[start_context_idx])
    print(start_context)

    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        start_context=start_context,
        tokenizer=tokenizer,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        checkpoint_path=args.checkpoint,
        lr=args.lr,
        generate_max_new_tokens=256,
        generate_context_size=args.allowed_max_length,
        generate_eos_id=tokenizer.eos_token_id or 151643,
        patience=args.patience,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


if __name__ == "__main__":
    main()
