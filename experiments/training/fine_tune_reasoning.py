"""
Fine-tuning script for CoreOutline Qwen model on financial reasoning data.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core import CoreOutlineConfig, create_coreoutline_qwen_model
from src.training.reasoning_data import format_reasoning_input, load_reasoning_data
from src.training.trainer import train_model_simple


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CoreOutline on financial reasoning data")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="./data/finetuning_llm.csv",
        help="Path to the training CSV file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./models/nyx_2_reasoning.pth",
        help="Path to load/save model checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--eval-iter", type=int, default=2)
    parser.add_argument("--allowed-max-length", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
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

    train_loader, val_loader, test_loader, tokenizer, val_data = load_reasoning_data(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allowed_max_length=args.allowed_max_length,
        device=str(device),
    )

    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        print(f"Checkpoint {args.checkpoint} not found, training from scratch.")

    model.config.use_cache = False
    model.config.gradient_checkpointing = True

    start_context_idx = 10 if len(val_data) > 10 else 0
    start_context = format_reasoning_input(val_data.iloc[start_context_idx])
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
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


if __name__ == "__main__":
    main()
