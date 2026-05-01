import os
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from .data import InstructionDataset, instruction_collate_fn
from .evaluate import calculate_loss
from src.models.core_outline_model import create_coreoutline_qwen_model, CoreOutlineConfig


BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 2048
NUM_EPOCHS = 100
EVAL_FREQ = 5
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed training')
        return None, None, None, None

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return rank, world_size, local_rank, dist.get_world_size()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model(checkpoint_path: str = None, config: CoreOutlineConfig = None):
    model = create_coreoutline_qwen_model(config=config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'Loading model checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print('Checkpoint loaded successfully')
    else:
        print('Creating model from scratch')
    return model


def save_model(model: torch.nn.Module, save_path: str, rank: int):
    if rank == 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f'Model saved to {save_path}')


def create_data_loaders(train_data, val_data, test_data, tokenizer, rank, world_size):
    from .data import InstructionDataset, instruction_collate_fn
    from torch.utils.data import DataLoader

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)

    collate_fn = partial(instruction_collate_fn, device=f'cuda:{rank}')

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer, max_length=MAX_SEQ_LENGTH),
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer, max_length=MAX_SEQ_LENGTH),
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        InstructionDataset(test_data, tokenizer, max_length=MAX_SEQ_LENGTH),
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


def evaluate(model: torch.nn.Module, val_loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        loss = calculate_loss(model, batch, device)
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def train_model(
    model,
    train_loader,
    val_loader,
    train_sampler,
    val_sampler,
    device,
    rank,
    world_size,
    num_epochs: int = NUM_EPOCHS,
    accumulation_steps: int = ACCUMULATION_STEPS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    gradient_clip: float = GRADIENT_CLIP,
    eval_freq: int = EVAL_FREQ,
):
    model.to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.1)
    scaler = GradScaler()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', disable=rank != 0) if rank == 0 else train_loader
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            with autocast():
                loss = calculate_loss(model, batch, device)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps
            num_train_batches += 1

            if rank == 0:
                progress_bar.set_postfix({"Train Loss": f"{loss.item() * accumulation_steps:.4f}"})

        scheduler.step()
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('inf')

        if (epoch + 1) % eval_freq == 0:
            val_loss = evaluate(model, val_loader, device)
            if rank == 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model.module, f"./models/nyx_ddp_best.pth", rank)
            if rank == 0:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if (epoch + 1) % 10 == 0 and rank == 0:
            save_model(model.module, f"./models/nyx_ddp_epoch_{epoch+1}.pth", rank)

    if rank == 0:
        save_model(model.module, f"./models/nyx_ddp_final.pth", rank)

    return model
