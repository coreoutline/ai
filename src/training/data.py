import os
from functools import partial
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


DEFAULT_TOKENIZER = "Qwen/Qwen1.5-0.5B"
DEFAULT_PAD_TOKEN_ID = 50256
DEFAULT_IGNORE_INDEX = -100


def format_instruction(entry: pd.Series, include_response: bool = True) -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['prompts'].strip().replace('<prompt>', '').replace('</prompt>', '')}"
    )
    input_text = (
        f"\n\n### Input: \n{entry['input'].strip()}" if "input" in entry and entry["input"] else ""
    )
    if include_response:
        desired_response = f"\n\n### Response: \n{entry['answers'].strip().replace('<ans>', '').replace('</ans>', '')}"
        return instruction_text + input_text + desired_response
    return instruction_text + input_text


def format_input(entry: pd.Series) -> str:
    return format_instruction(entry, include_response=False)


class InstructionDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        entry = self.data.iloc[index]
        full_text = format_instruction(entry)
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def instruction_collate_fn(
    batch,
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    allowed_max_length: Optional[int] = None,
    device: str = "cpu"
):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    if allowed_max_length is not None:
        input_ids = input_ids[:, :allowed_max_length]
        attention_mask = attention_mask[:, :allowed_max_length]
        labels = labels[:, :allowed_max_length]

    labels[labels == pad_token_id] = ignore_index

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def load_instruction_data(
    parquet_path: str = "hf://datasets/DeividasM/financial-instruction-aq22/data/train-00000-of-00001.parquet",
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_length: int = 2048,
    batch_size: int = 8,
    num_workers: int = 4,
    allowed_max_length: Optional[int] = None,
    device: str = "cpu",
    split: Tuple[float, float, float] = (0.85, 0.10, 0.05),
):
    df = pd.read_parquet(parquet_path)
    df = df.rename(columns={"instruction": "prompts", "output": "answers"})

    if sum(split) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    train_portion = int(len(df) * split[0])
    val_portion = int(len(df) * split[1])

    train_data = df.iloc[:train_portion].reset_index(drop=True)
    val_data = df.iloc[train_portion:train_portion + val_portion].reset_index(drop=True)
    test_data = df.iloc[train_portion + val_portion:].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    collate_fn = partial(
        instruction_collate_fn,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=DEFAULT_IGNORE_INDEX,
        allowed_max_length=allowed_max_length,
        device=device,
    )

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer, max_length=max_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer, max_length=max_length),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        InstructionDataset(test_data, tokenizer, max_length=max_length),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, tokenizer, val_data


def load_instruction_splits(
    parquet_path: str = "hf://datasets/DeividasM/financial-instruction-aq22/data/train-00000-of-00001.parquet",
    tokenizer_name: str = DEFAULT_TOKENIZER,
    split: Tuple[float, float, float] = (0.85, 0.10, 0.05),
):
    df = pd.read_parquet(parquet_path)
    df = df.rename(columns={"instruction": "prompts", "output": "answers"})
    if sum(split) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    train_portion = int(len(df) * split[0])
    val_portion = int(len(df) * split[1])

    train_data = df.iloc[:train_portion].reset_index(drop=True)
    val_data = df.iloc[train_portion:train_portion + val_portion].reset_index(drop=True)
    test_data = df.iloc[train_portion + val_portion:].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return train_data, val_data, test_data, tokenizer


def load_local_instruction_data(
    csv_path: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_length: int = 2048,
    batch_size: int = 8,
    num_workers: int = 4,
    allowed_max_length: Optional[int] = None,
    device: str = "cpu",
    split: Tuple[float, float, float] = (0.85, 0.10, 0.05),
):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"instruction": "prompts", "output": "answers"})
    if sum(split) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    train_portion = int(len(df) * split[0])
    val_portion = int(len(df) * split[1])

    train_data = df.iloc[:train_portion].reset_index(drop=True)
    val_data = df.iloc[train_portion:train_portion + val_portion].reset_index(drop=True)
    test_data = df.iloc[train_portion + val_portion:].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    collate_fn = partial(
        instruction_collate_fn,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=DEFAULT_IGNORE_INDEX,
        allowed_max_length=allowed_max_length,
        device=device,
    )

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer, max_length=max_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer, max_length=max_length),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        InstructionDataset(test_data, tokenizer, max_length=max_length),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, tokenizer, val_data
