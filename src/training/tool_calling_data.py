import json
from functools import partial
from typing import Any, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "Qwen/Qwen1.5-0.5B"
DEFAULT_PAD_TOKEN_ID = 50256
DEFAULT_IGNORE_INDEX = -100


def _parse_json_field(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Expected JSON string or list, got {type(value)!r}")


def _format_tools(tools: Any) -> str:
    parsed = _parse_json_field(tools)
    return "\n".join(json.dumps(tool, ensure_ascii=False) for tool in parsed)


def _format_answers(answers: Any) -> str:
    parsed = _parse_json_field(answers)
    return "\n".join(json.dumps(answer, ensure_ascii=False) for answer in parsed)


def format_tool_calling_prompt(entry: pd.Series) -> str:
    """Prompt-only text used for generation during evaluation."""
    query = entry["query"].strip()
    tools = _format_tools(entry["tools"])
    return (
        "You are a function-calling assistant. Given a user query and available tools, "
        "select the correct tool(s) and arguments.\n\n"
        f"### Query:\n{query}\n\n"
        f"### Available Tools:\n{tools}\n\n"
        "### Function Calls:\n"
    )


def format_tool_calling_example(entry: pd.Series, eos_token: str = "") -> str:
    """Full supervised example: prompt + expected function calls."""
    answers = _format_answers(entry["answers"])
    return format_tool_calling_prompt(entry) + answers + eos_token


def split_dataframe(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.85, 0.10, 0.05),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sum(split) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    train_portion = int(len(df) * split[0])
    val_portion = int(len(df) * split[1])

    train_data = df.iloc[:train_portion].reset_index(drop=True)
    val_data = df.iloc[train_portion : train_portion + val_portion].reset_index(drop=True)
    test_data = df.iloc[train_portion + val_portion :].reset_index(drop=True)
    return train_data, val_data, test_data


class EncodedToolCallingDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer):
        self.data = data.reset_index(drop=True)
        self.encoded_samples = []
        eos_token = tokenizer.eos_token or ""

        for _, entry in data.iterrows():
            prompt_text = format_tool_calling_prompt(entry)
            full_text = format_tool_calling_example(entry, eos_token=eos_token)
            prompt_ids = tokenizer.encode(prompt_text)
            full_ids = tokenizer.encode(full_text)
            self.encoded_samples.append(
                {"ids": full_ids, "prompt_len": len(prompt_ids)}
            )

    def __getitem__(self, index: int):
        return self.encoded_samples[index]

    def __len__(self) -> int:
        return len(self.data)


def tool_calling_collate_fn(
    batch,
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    allowed_max_length: Optional[int] = None,
    device: str = "cpu",
):
    batch_max_length = max(len(item["ids"]) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        token_ids = item["ids"].copy()
        prompt_len = item["prompt_len"]
        token_ids += [pad_token_id]
        padded = token_ids + [pad_token_id] * (batch_max_length - len(token_ids))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Mask padding tokens (keep loss on first pad only for EOS boundary).
        pad_mask = targets == pad_token_id
        pad_indices = torch.nonzero(pad_mask).squeeze()
        if pad_indices.numel() > 1:
            targets[pad_indices[1:]] = ignore_index

        # Mask prompt tokens so loss is computed only on function calls.
        answer_start = max(prompt_len - 1, 0)
        if answer_start > 0:
            targets[:answer_start] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)
    return inputs_tensor.to(device), targets_tensor.to(device)


def load_tool_calling_data(
    csv_path: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    batch_size: int = 4,
    num_workers: int = 0,
    allowed_max_length: Optional[int] = 2048,
    device: str = "cpu",
    split: Tuple[float, float, float] = (0.85, 0.10, 0.05),
    max_samples: Optional[int] = None,
):
    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.iloc[:max_samples].reset_index(drop=True)

    train_data, val_data, test_data = split_dataframe(df, split=split)

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    collate_fn = partial(
        tool_calling_collate_fn,
        pad_token_id=tokenizer.pad_token_id or DEFAULT_PAD_TOKEN_ID,
        ignore_index=DEFAULT_IGNORE_INDEX,
        allowed_max_length=allowed_max_length,
        device=device,
    )

    train_loader = DataLoader(
        EncodedToolCallingDataset(train_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        EncodedToolCallingDataset(val_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        EncodedToolCallingDataset(test_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, tokenizer, val_data
