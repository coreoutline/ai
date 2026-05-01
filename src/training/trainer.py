import torch
from torch import nn
from src.training.evaluate import evaluate_model, calc_loss_batch
import time
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from functools import partial


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
            input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader, start_context, tokenizer, num_epochs=2, eval_freq=5, eval_iter=2):
    # Initialize wandb
    # wandb.init(
    #     project="nyx-finetune",
    #     name="nyx-finetune-run"
    # )
    # wandb.watch(model, log="all", log_freq=10)

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=6e-6)
    print(scheduler.get_last_lr()[0])
    curr_loss = None
    model.to(device)
    for epoch in range(num_epochs):

        print(f"Epoch {epoch+1} training start...")
        model.train()
        for batch in train_loader:
            if isinstance(batch, dict):
                input_batch = batch["input_ids"]
                target_batch = batch["labels"]
            else:
                input_batch, target_batch = batch

            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # Generate example output for logging
                model.eval()
                example_input = start_context
                inputs = tokenizer(start_context, return_tensors="pt")
                from src.inference.generate import generate
                token_ids = generate(
                    model=model,
                    idx=inputs['input_ids'].to("cpu"),
                    max_new_tokens=30,
                    context_size=5012,
                    eos_id=50256,
                )

                print(tokenizer.decode(token_ids[0]))
                model.train()
                # Log metrics and example output to wandb
                # wandb.log({
                #     "epoch": epoch + 1,
                #     "global_step": global_step,
                #     "train_loss": train_loss,
                #     "val_loss": val_loss,
                #     "tokens_seen": tokens_seen,
                #     "learning_rate": scheduler.get_last_lr()[0]
                # }, step=global_step)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

            curr_loss = loss.item()
            torch.save(model.state_dict(), f"./models/nyx_2.pth")
            # torch.save(model.state_dict(), f"/root/transformers/models/nyx_2.pth")
            # Log model checkpoint to wandb

    return train_losses, val_losses, track_tokens_seen

