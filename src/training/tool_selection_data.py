"""Data loading for the zero-shot tool-selection (BART-MNLI) classifier.

Source: data/tool-use-multiturn-expanded.csv
Columns used:
    conversation_so_far  JSON list of {"from", "value"} turns -> the NLI premise
    tool_list            JSON list of {"function": {"name", "description"}}
    selected_tools       JSON list of tool names that were actually called

Each row has a variable number of candidate tools (1-8 in the source data), so
each row expands into one (premise, tool_description) NLI pair per candidate
tool. Flattening this way, rather than keeping one dataset item per row, is
required: PyTorch's default collate function needs every item in a batch to
have the same shape, and a per-row item would have a length equal to that
row's candidate count, which varies row to row.
"""
import json
from functools import partial
from typing import Any, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "facebook/bart-large-mnli"
DEFAULT_MAX_LENGTH = 512

# NLI label ids used by bart-large-mnli: 0=contradiction, 1=neutral, 2=entailment.
CONTRADICTION_LABEL = 0
ENTAILMENT_LABEL = 2


def _parse_json_field(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Expected JSON string or list, got {type(value)!r}")


def _conversation_to_premise(conversation_so_far: Any) -> str:
    turns = _parse_json_field(conversation_so_far)
    return " ".join(turn["value"] for turn in turns)


def prepare_tool_selection_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add the flattened premise/description/name columns the dataset needs."""
    df = df.copy()
    df["conversation_so_far_str"] = df["conversation_so_far"].apply(_conversation_to_premise)
    tool_lists = df["tool_list"].apply(_parse_json_field)
    df["tool_descriptions"] = tool_lists.apply(lambda tools: [t["function"]["description"] for t in tools])
    df["tool_names"] = tool_lists.apply(lambda tools: [t["function"]["name"] for t in tools])
    return df


def split_dataframe(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.85, 0.10, 0.05),
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sum(split) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    train_data = df.sample(frac=split[0], random_state=random_state)
    remainder = df.drop(train_data.index)
    val_frac = split[1] / (split[1] + split[2]) if (split[1] + split[2]) > 0 else 0.0
    val_data = remainder.sample(frac=val_frac, random_state=random_state)
    test_data = remainder.drop(val_data.index)

    return (
        train_data.reset_index(drop=True),
        val_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )


class ToolSelectionDataset(Dataset):
    """One (premise, tool_description) NLI pair per candidate tool."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.premises = df["conversation_so_far_str"].tolist()
        self.tool_names = df["tool_names"].tolist()
        self.tool_descriptions = df["tool_descriptions"].tolist()
        self.selected_tools = df["selected_tools"].apply(_parse_json_field).tolist()

        self.index = [
            (row, tool_idx)
            for row in range(len(self.df))
            for tool_idx in range(len(self.tool_descriptions[row]))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        row, tool_idx = self.index[idx]
        tool_name = self.tool_names[row][tool_idx]
        label = ENTAILMENT_LABEL if tool_name in self.selected_tools[row] else CONTRADICTION_LABEL

        enc = self.tokenizer(
            self.premises[row],
            self.tool_descriptions[row][tool_idx],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        # Squeeze the tokenizer's batch dim so every sample is [seq_len],
        # letting default_collate stack them into [batch, seq_len].
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return enc, torch.tensor(label, dtype=torch.long)


def tool_selection_collate_fn(batch, device: str = "cpu"):
    encodings, labels = zip(*batch)
    input_ids = torch.stack([e["input_ids"] for e in encodings]).to(device)
    attention_mask = torch.stack([e["attention_mask"] for e in encodings]).to(device)
    labels = torch.stack(labels).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, labels


def load_tool_selection_data(
    csv_path: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    batch_size: int = 16,
    num_workers: int = 0,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str = "cpu",
    split: Tuple[float, float, float] = (0.85, 0.10, 0.05),
    max_samples: Optional[int] = None,
):
    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.iloc[:max_samples].reset_index(drop=True)
    df = prepare_tool_selection_dataframe(df)

    train_data, val_data, test_data = split_dataframe(df, split=split)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataset = ToolSelectionDataset(train_data, tokenizer, max_length=max_length)
    val_dataset = ToolSelectionDataset(val_data, tokenizer, max_length=max_length)
    test_dataset = ToolSelectionDataset(test_data, tokenizer, max_length=max_length)

    print(f"Training rows: {len(train_data)} -> {len(train_dataset)} (premise, tool) pairs")
    print(f"Validation rows: {len(val_data)} -> {len(val_dataset)} (premise, tool) pairs")
    print(f"Test rows: {len(test_data)} -> {len(test_dataset)} (premise, tool) pairs")

    collate_fn = partial(tool_selection_collate_fn, device=device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, tokenizer, val_data
