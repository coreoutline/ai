import sys
import torch
import torch.nn as nn
import time
from torch.utils.data import random_split
from transformers import AutoTokenizer

sys.path.insert(0, '../../src')
sys.path.insert(0, '../..')

from src.core import create_coreoutline_qwen_model, CoreOutlineConfig
from src.training.trainer import train_model_simple
from src.training.preprocess_transformer_data import create_dataloader_v1
import wandb

def main():
    start_time = time.time()
    torch.manual_seed(123)
    
    wandb.init(project="chaos")

    with open("../../../data/code_contents.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        midpoint = len(raw_text) // 10
        raw_text = raw_text[:midpoint]

        print("Raw text loaded, length:")
        raw_text = raw_text.replace("<|endoftext|>", " ")
        dataloader, tokenizer = create_dataloader_v1(raw_text, batch_size=32, max_len=50, step=1, shuffle=False)
        total_size = len(dataloader.dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        print("Finished creating main data loader")
        train_dataset, val_dataset, test_dataset = random_split(dataloader.dataset, [train_size, val_size, test_size])
        print("Now splitting datasets")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("Finished preprocessing data and creating dataloaders.")

    config = CoreOutlineConfig(
        vocab_size=151936,
        hidden_size=256,            # much smaller than 1536
        intermediate_size=1024,     # 4x hidden size
        num_hidden_layers=6,        # reduced from 28
        num_attention_heads=4,      # 256 / 64 = 4
        num_key_value_heads=4,      # match heads
        max_position_embeddings=2048,  # typical for lightweight models
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0   ,
    )

    # Create the model
    core_model = create_coreoutline_qwen_model(config)

    with torch.no_grad():
        print(f"Model parameters: {sum(p.numel() for p in core_model.parameters()):,}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    
    start_context = "snowflake.connector"
    
    train_model_simple(
        model=core_model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        start_context=start_context,
        tokenizer=tokenizer,
        num_epochs=10,
        eval_freq=10,
        eval_iter=2
    )

if __name__ == "__main__":
    main()