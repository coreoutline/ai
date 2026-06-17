import torch
from typing import Any
from src.inference.generate import generate


def generate_and_print_sample(model: torch.nn.Module, tokenizer: Any, device: torch.device, start_context: str):
    model.eval()
    inputs = tokenizer(start_context, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    context_size = input_ids.shape[1]

    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=50,
            context_size=context_size,
        )

    decoded_text = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
    print(decoded_text.replace("\n", " "))
    model.train()


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    raise ValueError("Model output must be a tensor or dict containing 'logits'")


def calculate_loss(model: torch.nn.Module, batch, device: torch.device) -> torch.Tensor:
    if isinstance(batch, dict):
        input_batch = batch["input_ids"]
        target_batch = batch["labels"]
    else:
        input_batch, target_batch = batch
    return calc_loss_batch(input_batch, target_batch, model, device)


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    output = model(input_batch)
    logits = _extract_logits(output)

    if logits.ndim == 3:
        logits = logits.flatten(0, 1)
        target_batch = target_batch.flatten()
    loss = torch.nn.functional.cross_entropy(logits, target_batch, ignore_index=-100)
    return loss


def calc_loss_loader(data_loader, model: torch.nn.Module, device: torch.device, num_batches: int = None) -> float:
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        if isinstance(batch, dict):
            input_batch = batch["input_ids"]
            target_batch = batch["labels"]
        else:
            input_batch, target_batch = batch
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model: torch.nn.Module, train_loader, val_loader, device: torch.device, eval_iter: int):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
