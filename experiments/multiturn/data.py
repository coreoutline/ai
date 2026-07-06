"""Multi-turn dataset: renders conversations into a single supervised sequence.

The whole conversation becomes one token stream with a light chat template.
**Only assistant (`gpt`) turns are supervised** (LM loss); system / human / tool
turns are context (labels = -100). Every token also carries an **eval-only**
segment id (THINK / TOOL / RESPOND / DONE) derived from the assistant tags, used
to measure whether the latent gate rediscovers the modes in a real multi-turn
setting.

The end-of-turn marker after an assistant turn is supervised and labeled DONE, so
the gate learns a mode-level stop signal per turn.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.multiturn.parse import (
    SEG_DONE,
    SEG_IGNORE,
    parse_conversations,
    parse_tools,
    segment_assistant,
)

LABEL_IGNORE = -100

ROLE_PREFIX = {
    "system": "<|system|>\n",
    "human": "<|user|>\n",
    "gpt": "<|assistant|>\n",
    "tool": "<|tool|>\n",
}
END_OF_TURN = "<|end|>\n"


def _encode_conversation(messages, tokenizer, max_length: int) -> Optional[Dict]:
    eot_ids = tokenizer.encode(END_OF_TURN, add_special_tokens=False)
    input_ids: List[int] = []
    labels: List[int] = []
    segments: List[int] = []

    def add(ids, supervise, seg):
        input_ids.extend(ids)
        labels.extend(ids if supervise else [LABEL_IGNORE] * len(ids))
        segments.extend([seg] * len(ids) if supervise else [SEG_IGNORE] * len(ids))

    n_assistant = 0
    for m in messages:
        role = m["from"]
        prefix = ROLE_PREFIX.get(role, f"<|{role}|>\n")
        add(tokenizer.encode(prefix, add_special_tokens=False), False, SEG_IGNORE)

        if role == "gpt":
            n_assistant += 1
            for text, seg in segment_assistant(m["value"]):
                ids = tokenizer.encode(text, add_special_tokens=False)
                add(ids, True, seg)
            add(eot_ids, True, SEG_DONE)  # supervised stop signal
        else:
            add(tokenizer.encode(m["value"], add_special_tokens=False), False, SEG_IGNORE)
            add(eot_ids, False, SEG_IGNORE)

    if n_assistant == 0 or len(input_ids) < 2:
        return None

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        segments = segments[:max_length]
        if all(l == LABEL_IGNORE for l in labels):  # truncated away every target
            return None

    return {"input_ids": input_ids, "labels": labels, "segment_ids": segments}


def build_examples(
    tokenizer,
    csv_path: str,
    max_length: int = 2048,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Tokenized, segment-tagged training examples (one per conversation)."""
    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.iloc[:max_samples]
    examples = []
    for _, row in df.iterrows():
        messages = parse_conversations(row["conversations"])
        if not messages:
            continue
        enc = _encode_conversation(messages, tokenizer, max_length)
        if enc is not None:
            examples.append(enc)
    return examples


def build_eval_records(csv_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Structured records for teacher-forced per-turn evaluation.

    Each record: {messages, tools}. Kept separate from training tensors so the
    evaluator can regenerate assistant turns and score tool calls.
    """
    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.iloc[:max_samples]
    records = []
    for _, row in df.iterrows():
        messages = parse_conversations(row["conversations"])
        if not messages:
            continue
        records.append({"messages": messages, "tools": parse_tools(row.get("tools"))})
    return records


class MultiTurnDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def collate(batch: List[Dict], pad_id: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, segments, attn = [], [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [LABEL_IGNORE] * pad)
        segments.append(b["segment_ids"] + [SEG_IGNORE] * pad)
        attn.append([1] * n + [0] * pad)
    t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
    return {
        "input_ids": t(input_ids, torch.long),
        "labels": t(labels, torch.long),
        "segment_ids": t(segments, torch.long),
        "attention_mask": t(attn, torch.long),
    }


def split_examples(examples, val_fraction=0.1, test_fraction=0.05, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(examples), generator=g).tolist()
    examples = [examples[i] for i in perm]
    n = len(examples)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    return examples[n_test + n_val:], examples[n_test:n_test + n_val], examples[:n_test]


def render_context(messages: List[Dict], upto: int) -> str:
    """Render messages [0, upto) as chat context ending at the assistant prefix."""
    parts = []
    for m in messages[:upto]:
        parts.append(ROLE_PREFIX.get(m["from"], f"<|{m['from']}|>\n") + m["value"] + END_OF_TURN)
    parts.append(ROLE_PREFIX["gpt"])
    return "".join(parts)
