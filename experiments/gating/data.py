"""MixedModeDataset: unifies reasoning, tool-calling and plain-response data.

Every example is built as ``prompt + completion + eos`` where the completion is
split into segments that correspond to behavioral modes. Each token gets:

    input_ids      - the token id
    labels         - token id on the completion, -100 on prompt/pad (SFT masking)
    segment_ids    - THINK/TOOL/RESPOND/DONE on the completion, -100 elsewhere
    attention_mask - 1 on real tokens, 0 on pad

The ``segment_ids`` are **eval-only** ground truth for the mode-alignment probe.
They are *not* used by the model or the loss; training routing stays latent.
"""

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.gating.config import (
    SEGMENT_IGNORE,
    SEG_THINK,
    SEG_TOOL,
    SEG_RESPOND,
    SEG_DONE,
)

LABEL_IGNORE = -100


# --------------------------------------------------------------------------- #
# Section builders. Each returns a list of (text, segment_id, supervise) chunks.
# supervise=False -> prompt scaffold (no LM loss, no segment label).
# --------------------------------------------------------------------------- #
def _reasoning_chunks(row: pd.Series) -> List[tuple]:
    context = f"\n\n### Input:\n{str(row['context']).strip()}" if "context" in row and pd.notna(row.get("context")) else ""
    prompt = (
        "You are an expert financial / data / business analyst. Reason through the "
        "question, then answer.\n\n"
        f"### Instruction:\n{str(row['prompt']).strip()}"
        f"{context}"
        "\n\n### Reasoning:\n"
    )
    return [
        (prompt, SEGMENT_IGNORE, False),
        (str(row["thinking"]).strip(), SEG_THINK, True),
        ("\n\n### Response:\n", SEG_RESPOND, True),
        (str(row["answer"]).strip().replace("<answer>", "").replace("</answer>", ""), SEG_RESPOND, True),
    ]


def _tool_chunks(row: pd.Series) -> List[tuple]:
    tools = _stringify_json(row["tools"])
    answers = _stringify_json(row["answers"])
    prompt = (
        "You are a function-calling assistant. Given a user query and available "
        "tools, select the correct tool(s) and arguments.\n\n"
        f"### Query:\n{str(row['query']).strip()}\n\n"
        f"### Available Tools:\n{tools}\n\n"
        "### Function Calls:\n"
    )
    return [
        (prompt, SEGMENT_IGNORE, False),
        (answers, SEG_TOOL, True),
    ]


def _plain_chunks(row: pd.Series) -> List[tuple]:
    prompt = (
        "You are a helpful financial assistant. Answer the user's question directly.\n\n"
        f"### Instruction:\n{str(row['prompts']).strip()}\n\n### Response:\n"
    )
    return [
        (prompt, SEGMENT_IGNORE, False),
        (str(row["answers"]).strip(), SEG_RESPOND, True),
    ]


def _stringify_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value.strip()
    if isinstance(value, list):
        return "\n".join(json.dumps(v, ensure_ascii=False) for v in value)
    return json.dumps(value, ensure_ascii=False)


def _encode_example(chunks: List[tuple], tokenizer, eos_id: int, max_length: int) -> Optional[Dict]:
    input_ids: List[int] = []
    labels: List[int] = []
    segment_ids: List[int] = []

    for text, seg, supervise in chunks:
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(ids)
        if supervise:
            labels.extend(ids)
            segment_ids.extend([seg] * len(ids))
        else:
            labels.extend([LABEL_IGNORE] * len(ids))
            segment_ids.extend([SEGMENT_IGNORE] * len(ids))

    # Terminal EOS carries the DONE segment (the learned stop signal target).
    input_ids.append(eos_id)
    labels.append(eos_id)
    segment_ids.append(SEG_DONE)

    if len(input_ids) < 2 or len(input_ids) > max_length:
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            segment_ids = segment_ids[:max_length]
        else:
            return None

    return {"input_ids": input_ids, "labels": labels, "segment_ids": segment_ids}


def build_mixed_examples(
    tokenizer,
    reasoning_csv: Optional[str] = None,
    tool_csv: Optional[str] = None,
    plain_csv: Optional[str] = None,
    max_length: int = 1024,
    max_samples_per_source: Optional[int] = None,
) -> List[Dict]:
    """Read the source CSVs and return tokenized, segment-tagged examples."""
    eos_id = tokenizer.eos_token_id
    examples: List[Dict] = []

    sources = [
        (reasoning_csv, _reasoning_chunks),
        (tool_csv, _tool_chunks),
        (plain_csv, _plain_chunks),
    ]
    for csv_path, chunk_fn in sources:
        if not csv_path:
            continue
        df = pd.read_csv(csv_path)
        if max_samples_per_source is not None:
            df = df.iloc[:max_samples_per_source]
        for _, row in df.iterrows():
            try:
                enc = _encode_example(chunk_fn(row), tokenizer, eos_id, max_length)
            except (KeyError, TypeError, ValueError):
                continue
            if enc is not None:
                enc["source"] = chunk_fn.__name__
                examples.append(enc)
    return examples


class MixedModeDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def collate_mixed(batch: List[Dict], pad_id: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, segment_ids, attn = [], [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [LABEL_IGNORE] * pad)
        segment_ids.append(b["segment_ids"] + [SEGMENT_IGNORE] * pad)
        attn.append([1] * n + [0] * pad)
    to = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
    return {
        "input_ids": to(input_ids, torch.long),
        "labels": to(labels, torch.long),
        "segment_ids": to(segment_ids, torch.long),
        "attention_mask": to(attn, torch.long),
    }


def split_examples(examples, val_fraction=0.1, test_fraction=0.05, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(examples), generator=g).tolist()
    examples = [examples[i] for i in perm]
    n = len(examples)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test = examples[:n_test]
    val = examples[n_test : n_test + n_val]
    train = examples[n_test + n_val :]
    return train, val, test
