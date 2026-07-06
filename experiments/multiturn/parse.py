"""Robust parsing of the ToolAce/Hermes multi-turn ``conversations`` field.

The CSV stores each conversation as a Python-literal-ish string, but with raw
newlines used *both* inside string values and as structural separators between
message dicts (so ``ast.literal_eval`` fails). Rather than repair the literal, we
extract messages by their unambiguous boundary marker ``{'from': '`` — that
marker never appears inside a value — which is robust to nested quotes, tags, and
embedded JSON.

Each conversation is a list of ``{"from": role, "value": text}`` where role is
one of: ``system`` (instructions + <tools>), ``human`` (user), ``gpt``
(assistant: <think>/<tool_call>/response), ``tool`` (<tool_response> result).

We also expose segment tagging that maps assistant text onto the gate's modes:
THINK / TOOL / RESPOND / DONE.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Mode ids mirror src.core.gating.Mode (kept local to avoid a hard import here).
SEG_IGNORE = -100
SEG_THINK = 0
SEG_TOOL = 1
SEG_RESPOND = 2
SEG_DONE = 3
SEGMENT_NAMES = ["THINK", "TOOL", "RESPOND", "DONE"]

_MSG_START = re.compile(r"\{'from':\s*'")
_ROLE_RE = re.compile(r"\{'from':\s*'([^']*)'\s*,\s*'value':\s*'")

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOLCALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _unescape(val: str) -> str:
    return (
        val.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\'", "'")
        .replace('\\"', '"')
    )


def parse_conversations(raw: str) -> List[Dict[str, str]]:
    """Parse the ``conversations`` cell into a list of role/value messages."""
    if not isinstance(raw, str):
        return []
    starts = [m.start() for m in _MSG_START.finditer(raw)]
    msgs: List[Dict[str, str]] = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(raw)
        block = raw[s:end]
        rm = _ROLE_RE.match(block)
        if not rm:
            continue
        role = rm.group(1)
        val = block[rm.end():].rstrip()
        for suf in ["'}],", "'}]", "'},", "'}"]:
            if val.endswith(suf):
                val = val[: -len(suf)]
                break
        msgs.append({"from": role, "value": _unescape(val)})
    return msgs


def parse_tools(raw: Any) -> List[Dict[str, Any]]:
    """Parse the ``tools`` column (JSON list of function signatures)."""
    if isinstance(raw, list):
        tools = raw
    elif isinstance(raw, str):
        try:
            tools = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        return []
    out = []
    for t in tools:
        if isinstance(t, dict) and t.get("name"):
            out.append(t)
    return out


def segment_assistant(value: str) -> List[Tuple[str, int]]:
    """Split an assistant turn into ordered (text, segment_id) chunks.

    <think>...</think> -> THINK, <tool_call>...</tool_call> -> TOOL, everything
    else (the natural-language answer) -> RESPOND. Tags are kept inside their
    segment so the model learns to emit them.
    """
    spans: List[Tuple[int, int, int]] = []  # (start, end, seg)
    for m in _THINK_RE.finditer(value):
        spans.append((m.start(), m.end(), SEG_THINK))
    for m in _TOOLCALL_RE.finditer(value):
        spans.append((m.start(), m.end(), SEG_TOOL))
    spans.sort()

    chunks: List[Tuple[str, int]] = []
    cursor = 0
    for start, end, seg in spans:
        if start > cursor:
            text = value[cursor:start]
            if text.strip():
                chunks.append((text, SEG_RESPOND))
        chunks.append((value[start:end], seg))
        cursor = end
    if cursor < len(value):
        tail = value[cursor:]
        if tail.strip():
            chunks.append((tail, SEG_RESPOND))
    if not chunks and value.strip():
        chunks.append((value, SEG_RESPOND))
    return chunks


def extract_tool_calls(value: str) -> List[Dict[str, Any]]:
    """Extract structured function calls from an assistant turn's <tool_call>s."""
    calls = []
    for m in _TOOLCALL_RE.finditer(value):
        body = m.group(1).strip()
        parsed = _loads_loose(body)
        if isinstance(parsed, dict):
            calls.append(parsed)
    return calls


def _loads_loose(body: str) -> Optional[Any]:
    """Parse a tool-call body that may use single quotes (Python-style)."""
    for attempt in (body, body.replace("'", '"')):
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
    try:
        import ast

        return ast.literal_eval(body)
    except (ValueError, SyntaxError):
        return None


def last_user_query(messages: List[Dict[str, str]], upto: int) -> str:
    """Most recent human message before assistant-turn index ``upto``."""
    for j in range(upto - 1, -1, -1):
        if messages[j]["from"] == "human":
            return messages[j]["value"]
    return ""
