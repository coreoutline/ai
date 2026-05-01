import sys
import os
sys.path.insert(0, '../..')

import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.core import create_coreoutline_qwen_model, CoreOutlineConfig
from src.training.trainer import train_model_simple
from src.training.data import load_instruction_data, format_input

def main():
    torch.manual_seed(123)
    
    # 1. Create Model
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
    core_model = create_coreoutline_qwen_model(config)
    
    with torch.no_grad():
        print(f"Model parameters: {sum(p.numel() for p in core_model.parameters()):,}")

    # 2. Load Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # data.py load_instruction_data returns: train_loader, val_loader, test_loader, tokenizer, val_data
    train_loader, val_loader, test_loader, tokenizer, val_data = load_instruction_data(
        parquet_path="hf://datasets/DeividasM/financial-instruction-aq22/data/train-00000-of-00001.parquet",
        tokenizer_name="Qwen/Qwen1.5-0.5B",
        max_length=2048,
        batch_size=8,
        num_workers=4,
        allowed_max_length=10240, 
        device=str(device)
    )

    # 3. Load Model Weights
    model_path = "./models/nyx_2.pth"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model_state_dict = torch.load(model_path, map_location="cpu")
        core_model.load_state_dict(model_state_dict)
    else:
        print(f"File {model_path} not found, initializing from scratch.")
        
    core_model.to("cpu")
    core_model.eval()

    # Get sample inputs for tracking output logic
    torch.manual_seed(123)
    input_text = format_input(val_data.iloc[10]) if len(val_data) > 10 else format_input(val_data.iloc[0])
    print(input_text)
    start_context = format_input(val_data.iloc[0])

    # 4. Train Model
    start_time = time.time()
    torch.manual_seed(123)
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=core_model,
        train_loader=train_loader,
        val_loader=val_loader,
        start_context=start_context,
        tokenizer=tokenizer,
        num_epochs=2,
        eval_freq=5,
        eval_iter=2
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

if __name__ == "__main__":
    main()
