import torch
from tqdm import tqdm

from src.training.evaluate import evaluate_model, calc_loss_batch
from src.inference.generate import generate


def train_model_simple(
    model,
    train_loader,
    val_loader,
    start_context,
    tokenizer,
    *,
    num_epochs=100,
    eval_freq=5,
    eval_iter=2,
    checkpoint_path="./models/nyx_2.pth",
    lr=6e-5,
    weight_decay=0.1,
    scheduler_t_max=50,
    scheduler_eta_min=6e-6,
    generate_max_new_tokens=100,
    generate_context_size=5012,
    generate_eos_id=50256,
    generate_temperature=0.7,
    generate_top_k=50,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_t_max, eta_min=scheduler_eta_min
    )
    print(scheduler.get_last_lr()[0])
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} training start...")
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if isinstance(batch, dict):
                input_batch = batch["input_ids"]
                target_batch = batch["labels"]
            else:
                input_batch, target_batch = batch

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                model.eval()
                inputs = tokenizer(start_context, return_tensors="pt")
                token_ids = generate(
                    model=model,
                    idx=inputs["input_ids"].to(device),
                    max_new_tokens=generate_max_new_tokens,
                    context_size=generate_context_size,
                    eos_id=generate_eos_id,
                    temperature=generate_temperature,
                    top_k=generate_top_k,
                )
                print(tokenizer.decode(token_ids[0]))
                model.train()

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

            torch.save(model.state_dict(), checkpoint_path)
            model.train()

    return train_losses, val_losses, track_tokens_seen
