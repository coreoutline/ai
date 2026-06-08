from functools import partial
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "Qwen/Qwen1.5-0.5B"
DEFAULT_PAD_TOKEN_ID = 50256
DEFAULT_IGNORE_INDEX = -100


def format_reasoning_example(entry: pd.Series) -> str:
    instruction_text = (
        "You are an expert financial analyst / accountant / data analyst / business intelligence analyst\n\n"
        "You are required to reason through and answer customer questions providing them with answers to their questions. Here is the instruction: \n\n"
        f"\n\n ### Instruction:\n{entry['prompt'].strip()}"
    )
    input_text = (
        "Here is the context through which you will base your answer on: "
        f"\n\n### Input: \n{entry['context'].strip()}" if "context" in entry else ""
    )
    reasoning_step = (
        "\n\nBased on the instruction and the context, here are the reasoning steps that should be taken: "
        f"\n\n### Reasoning: \n{entry['thinking'].strip()}"
    )
    desired_response = (
        f"\n\n### Response: \n{entry['answer'].strip().replace('<answer>', '').replace('</answer>', '')}"
    )
    return instruction_text + input_text + reasoning_step + desired_response + "<|im_end|>"


def format_reasoning_input(entry: pd.Series) -> str:
    instruction_text = (
        "You are an expert financial analyst / accountant / data analyst / business intelligence analyst\n\n"
        "You are required to reason through and answer customer questions providing them with answers to their questions. Here is the instruction: \n\n"
        f"\n\n ### Instruction:\n{entry['prompt'].strip()}"
    )
    input_text = (
        "Here is the context through which you will base your answer on: "
        f"\n\n### Input: \n{entry['context'].strip()}" if "context" in entry else ""
    )
    reasoning_step = (
        "\n\nBased on the instruction and the context, here are the reasoning steps that should be taken: \n"
        "\n\n### Reasoning: \n"
    )
    return instruction_text + input_text + reasoning_step


def split_dataframe(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.85, 0.10, 0.05),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sum(split) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    train_portion = int(len(df) * split[0])
    val_portion = int(len(df) * split[1])

    train_data = df.iloc[:train_portion].reset_index(drop=True)
    val_data = df.iloc[train_portion:train_portion + val_portion].reset_index(drop=True)
    test_data = df.iloc[train_portion + val_portion:].reset_index(drop=True)
    return train_data, val_data, test_data


class EncodedReasoningDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer):
        self.data = data.reset_index(drop=True)
        self.encoded_texts = []
        for _, entry in data.iterrows():
            full_text = format_reasoning_example(entry)
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index: int):
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.data)


def shift_collate_fn(
    batch,
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    allowed_max_length: Optional[int] = None,
    device: str = "cpu",
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)
    return inputs_tensor.to(device), targets_tensor.to(device)


def load_reasoning_data(
    csv_path: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    batch_size: int = 4,
    num_workers: int = 0,
    allowed_max_length: Optional[int] = 4096,
    device: str = "cpu",
    split: Tuple[float, float, float] = (0.85, 0.10, 0.05),
):
    df = pd.read_csv(csv_path)
    train_data, val_data, test_data = split_dataframe(df, split=split)

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    collate_fn = partial(
        shift_collate_fn,
        pad_token_id=tokenizer.pad_token_id or DEFAULT_PAD_TOKEN_ID,
        ignore_index=DEFAULT_IGNORE_INDEX,
        allowed_max_length=allowed_max_length,
        device=device,
    )

    train_loader = DataLoader(
        EncodedReasoningDataset(train_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        EncodedReasoningDataset(val_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        EncodedReasoningDataset(test_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, tokenizer, val_data
