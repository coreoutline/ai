import torch
from torch import nn
from ray import tune
from src.training.evaluate import evaluate_model, calc_loss_batch

def train_model_ray(config, train_loader, val_loader, model_class, device=None, num_epochs=2, eval_freq=5, eval_iter=2):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_class(config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=0.1)
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} training start...")
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
                      f"Val loss {val_loss:.3f}")
                
            tune.report({"loss": loss.item()})
            
        torch.save(model.state_dict(), f"core_foundation_epoch_{epoch+1}.pth")
        
    return train_losses, val_losses, track_tokens_seen
