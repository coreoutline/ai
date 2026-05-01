"""
Fine-tuning script for CoreOutline Qwen model using DDP with torchrun compatibility
"""
import os
import sys
import torch

sys.path.insert(0, '../..')

from src.training.ddp import (
    setup_distributed, 
    cleanup_distributed, 
    train_model, 
    load_model,
    create_data_loaders
)
from src.training.data import load_instruction_splits

def main():
    rank, world_size, local_rank, _ = setup_distributed()
    if rank is None:
        print("This script should be run with torchrun")
        return
        
    print(f"Running on rank {rank}/{world_size-1} (local rank {local_rank})")
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    train_data, val_data, test_data, tokenizer = load_instruction_splits()
    
    train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = create_data_loaders(
        train_data, val_data, test_data, tokenizer, rank, world_size
    )
    
    model = load_model("./models/nyx_2.pth")
    model.config.use_cache = False
    model.config.gradient_checkpointing = True
    
    try:
        model = train_model(
            model, train_loader, val_loader, train_sampler, val_sampler, 
            device, rank, world_size
        )
        print("Training completed successfully")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()