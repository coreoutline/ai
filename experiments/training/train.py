import sys
sys.path.insert(0, '../..')

import torch
from src.training.evaluate import evaluate_model, generate_and_print_sample, calc_loss_batch
from src.models.model import CoreModel
from src.training.preprocess_transformer_data import create_dataloader_v1
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import time
start_time = time.time()
torch.manual_seed(123)
from src.config.config import CORE_TRANSFORMER_CONFIG
from torch.utils.data import random_split
from torch import nn

with open("../../../data/verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    raw_text = raw_text.replace("<|endoftext|>", " <|endoftext|> ")
    dataloader, tokenizer = create_dataloader_v1( raw_text, batch_size=1, max_len=100, step=1, shuffle=False)
    total_size = len(dataloader.dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataloader.dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

def train_model_simple(config):
    num_epochs=2
    eval_freq=5
    eval_iter=2
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    model =  CoreModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} training start...")
        model.to(device)
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
            input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, "
                f"Val loss {val_loss:.3f}"
                )
            tune.report({"loss":loss.item()})
        torch.save(model.state_dict(), f"core_foundation{epoch+1}.pth")
        # generate_and_print_sample(
        #     model, tokenizer, device, start_context
        # )
    return train_losses, val_losses, track_tokens_seen






    # data_iter = iter(dataloader)
    # first_batch = next(data_iter)
    # print(first_batch)
if __name__ == "__main__":
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=1,
        grace_period=1,
        reduction_factor=2
    )
    analysis = tune.run(
        train_model_simple,
        config=CORE_TRANSFORMER_CONFIG,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=tune.JupyterNotebookReporter(overwrite=True)
    )
# num_epochs = 2
# train_losses, val_losses, tokens_seen = train_model_simple( train_loader, val_loader, num_epochs=num_epochs, eval_freq=5, eval_iter=5, tokenizer=tokenizer,config=CORE_TRANSFORMER_CONFIG)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")