import sys
import os
import torch
import time
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import random_split
from functools import partial

# Ensure the root of the project is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.training.ray_trainer import train_model_ray
from src.core.base_transformer import CoreModel
from src.training.preprocess_transformer_data import create_dataloader_v1
from src.config.baseline_config import CORE_TRANSFORMER_CONFIG

def main():
    start_time = time.time()
    torch.manual_seed(123)

    # Load data from the relative path
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/verdict.txt"))
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please ensure the data directory exists.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        raw_text = raw_text.replace("<|endoftext|>", " <|endoftext|> ")
        dataloader, tokenizer = create_dataloader_v1(raw_text, batch_size=1, max_len=100, step=1, shuffle=False)
        
        total_size = len(dataloader.dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataloader.dataset, [train_size, val_size, test_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("Data loaded and dataloaders prepared.")

    # Initialize Ray Tune scheduler
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=1,
        grace_period=1,
        reduction_factor=2
    )

    # Prepare training function for Ray
    train_fn = partial(
        train_model_ray, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        model_class=CoreModel,
        num_epochs=1,  # Reduced for standard run
        eval_freq=5,
        eval_iter=2
    )

    print("Starting Ray Tune experiment...")
    # Run Ray Tune experiment
    analysis = tune.run(
        train_fn,
        config=CORE_TRANSFORMER_CONFIG,
        num_samples=1,
        scheduler=scheduler
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

if __name__ == "__main__":
    main()