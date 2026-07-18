"""
process_multiturn.py
====================
Processes tool-use-multiturn-reasoning-with-tool-list.csv:

1. Removes 'gpt' turns from conversations.
2. Strips the <tools>...</tools> block from the system prompt.
3. Expands each conversation into cumulative rows (one per human turn).
4. Adds three new columns:
   - conversation_so_far : accumulated conversation up to that human turn
   - tool_list           : the tool_list column value (available tools)
   - selected_tools      : JSON list of tool names called so far

Output: tool-use-multiturn-expanded.csv (same directory as input)
"""

import csv
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_FILE = Path(
    r"C:\Users\tsuma.thomas\Documents\CoreOutline\transformer\data"
    r"\tool-use-multiturn-reasoning-with-tool-list.csv"
)
OUTPUT_FILE = INPUT_FILE.parent / "tool-use-multiturn-expanded.csv"

# ---------------------------------------------------------------------------
# Regex to strip the <tools>…</tools> block from the system prompt.
# ---------------------------------------------------------------------------
TOOLS_BLOCK_RE = re.compile(
    r"Here are the available tools:\s*\n\s*<tools>\s*\n.*?</tools>\s*\n",
    re.DOTALL,
)


def strip_tools_from_system(text: str) -> str:
    """Remove the embedded <tools> block from a system prompt."""
    cleaned = TOOLS_BLOCK_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Tool-call extraction from a gpt turn value
# ---------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
TOOL_NAME_RE = re.compile(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']')


def extract_tool_names(gpt_value: str) -> list:
    """Return list of tool names called in a gpt turn value string."""
    names = []
    for match in TOOL_CALL_RE.finditer(gpt_value):
        raw = match.group(1)
        # Normalise doubled-quote artefacts from CSV storage
        normalised = raw.replace('""', '"')
        # Try JSON first
        try:
            obj = json.loads(normalised)
            if isinstance(obj, dict) and "name" in obj:
                names.append(obj["name"])
                continue
        except json.JSONDecodeError:
            pass
        # Regex fallback
        m = TOOL_NAME_RE.search(raw)
        if m:
            names.append(m.group(1))
    return names


# ---------------------------------------------------------------------------
# Conversation parsing
# ---------------------------------------------------------------------------
# The conversations column is a Python repr of a list of dicts like:
#   [{'from': 'system', 'value': 'text...'}, {'from': 'human', 'value': '...'}, ...]
#
# The value strings may contain:
#   - \n  (actual newlines, used as separators between list elements too)
#   - \\  (literal backslash)
#   - \'  (escaped single-quote inside single-quoted value strings)
#   - \"  (double-quoted substrings within the value)
#
# ast.literal_eval fails because:
#   - Actual \n characters between dict elements confuse the parser
#   - \' sequences break single-quoted strings
#
# We parse with a hand-written extractor that uses a state-machine to find
# all {'from': X, 'value': Y} dicts in the list.

# Pattern to find each turn: locate 'from' and 'value' keys.
# We'll use a targeted regex that handles both orderings of keys.

TURN_RE = re.compile(
    r"\{[^{}]*?"
    r"['\"]from['\"]\s*:\s*['\"](?P<role>[^'\"]+)['\"]"
    r"[^{}]*?"
    r"['\"]value['\"]\s*:\s*",
    re.DOTALL,
)


def _extract_single_quoted_value(s: str, start: int) -> tuple:
    """
    Extract a single-quoted Python string starting at index `start` in `s`.
    Handles \\' escapes.  Returns (value_str, end_index) or (None, -1).
    """
    if start >= len(s) or s[start] != "'":
        return None, -1

    i = start + 1
    parts = []
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            # Escape sequence
            if i + 1 < len(s):
                next_ch = s[i + 1]
                if next_ch == "'":
                    parts.append("'")
                    i += 2
                elif next_ch == "\\":
                    parts.append("\\")
                    i += 2
                elif next_ch == "n":
                    parts.append("\n")
                    i += 2
                elif next_ch == "t":
                    parts.append("\t")
                    i += 2
                elif next_ch == "r":
                    parts.append("\r")
                    i += 2
                else:
                    # Keep as-is
                    parts.append("\\")
                    parts.append(next_ch)
                    i += 2
            else:
                parts.append(ch)
                i += 1
        elif ch == "'":
            # End of string
            return "".join(parts), i + 1
        else:
            parts.append(ch)
            i += 1

    return None, -1  # Unterminated string


def _extract_quoted_value(s: str, start: int) -> tuple:
    """
    Extract a quoted value (single or double quote) starting at `start`.
    Returns (value_str, end_index) or (None, -1).
    """
    if start >= len(s):
        return None, -1

    quote = s[start]
    if quote == "'":
        return _extract_single_quoted_value(s, start)
    elif quote == '"':
        # Double-quoted string
        i = start + 1
        parts = []
        while i < len(s):
            ch = s[i]
            if ch == "\\":
                if i + 1 < len(s):
                    next_ch = s[i + 1]
                    if next_ch == '"':
                        parts.append('"')
                    elif next_ch == "\\":
                        parts.append("\\")
                    elif next_ch == "n":
                        parts.append("\n")
                    elif next_ch == "t":
                        parts.append("\t")
                    elif next_ch == "r":
                        parts.append("\r")
                    else:
                        parts.append("\\")
                        parts.append(next_ch)
                    i += 2
                else:
                    parts.append(ch)
                    i += 1
            elif ch == '"':
                return "".join(parts), i + 1
            else:
                parts.append(ch)
                i += 1
        return None, -1
    return None, -1


def parse_conversation(raw: str) -> list:
    """
    Parse the conversations column into a list of {'from': role, 'value': text} dicts.
    Returns [] on failure.
    """
    if not isinstance(raw, str) or not raw.strip().startswith("["):
        return []

    turns = []
    i = 0
    n = len(raw)

    while i < n:
        # Look for start of a dict '{'
        brace_pos = raw.find("{", i)
        if brace_pos == -1:
            break

        # From within this dict, find 'from' key and its value
        # Then find 'value' key and its value
        # We scan forward from brace_pos

        j = brace_pos + 1
        role = None
        value = None

        # Scan key-value pairs in this dict until we hit '}'
        while j < n and raw[j] != "}":
            # Skip whitespace and commas
            while j < n and raw[j] in " \t\n\r,":
                j += 1
            if j >= n or raw[j] == "}":
                break

            # Extract key
            key, j = _extract_quoted_value(raw, j)
            if key is None:
                j += 1
                continue

            # Skip ':'
            while j < n and raw[j] in " \t\n\r":
                j += 1
            if j < n and raw[j] == ":":
                j += 1
            while j < n and raw[j] in " \t\n\r":
                j += 1

            # Extract value
            if j < n and raw[j] in ("'", '"'):
                val, j = _extract_quoted_value(raw, j)
                if val is None:
                    # Extraction failed; skip to next comma or brace
                    while j < n and raw[j] not in (",", "}"):
                        j += 1
                    continue
            else:
                # Non-string value (e.g. None, True, False, number)
                # Read until comma or closing brace
                val_start = j
                depth = 0
                while j < n:
                    if raw[j] in ("{", "["):
                        depth += 1
                    elif raw[j] in ("}", "]"):
                        if depth == 0:
                            break
                        depth -= 1
                    elif raw[j] == "," and depth == 0:
                        break
                    j += 1
                val = raw[val_start:j].strip()

            if key == "from":
                role = val
            elif key == "value":
                value = val

        # Advance past the closing '}'
        close_brace = raw.find("}", j)
        if close_brace == -1:
            i = j
        else:
            i = close_brace + 1

        if role is not None and value is not None:
            turns.append({"from": role, "value": value})
        elif role is not None or value is not None:
            # Partial; skip
            pass

    return turns


# ---------------------------------------------------------------------------
# Main processing logic
# ---------------------------------------------------------------------------

def process_row(original_row: dict) -> list:
    """
    Given one original CSV row, return a list of expanded output rows –
    one per human turn.
    """
    raw_conv = original_row.get("conversations", "")
    if not isinstance(raw_conv, str):
        return []

    turns = parse_conversation(raw_conv)
    if not turns:
        return []

    tool_list_value = original_row.get("tool_list", "")

    # Separate system turn and clean it.
    system_turn = None
    non_system_turns = []
    for turn in turns:
        role = turn.get("from", "")
        if role == "system":
            cleaned_value = strip_tools_from_system(turn.get("value", ""))
            system_turn = {"from": "system", "value": cleaned_value}
        else:
            non_system_turns.append(turn)

    output_rows = []
    accumulated_turns = []

    for idx, turn in enumerate(non_system_turns):
        role = turn.get("from", "")
        if role == "human":
            # Build conversation_so_far including system + previous human/tool turns + current human turn
            conv_so_far = []
            if system_turn:
                conv_so_far.append(system_turn)
            conv_so_far.extend(accumulated_turns)
            conv_so_far.append(turn)

            # Look ahead for tools called in responding gpt turns before the next human turn
            selected_tools = []
            j = idx + 1
            while j < len(non_system_turns):
                next_turn = non_system_turns[j]
                next_role = next_turn.get("from", "")
                if next_role == "human":
                    break
                elif next_role == "gpt":
                    tool_names = extract_tool_names(next_turn.get("value", ""))
                    selected_tools.extend(tool_names)
                j += 1

            new_row = dict(original_row)
            new_row["conversation_so_far"] = json.dumps(conv_so_far, ensure_ascii=False)
            new_row["tool_list"] = tool_list_value
            new_row["selected_tools"] = json.dumps(selected_tools)
            output_rows.append(new_row)

            # Include this human turn in subsequent turns
            accumulated_turns.append(turn)
        elif role == "tool":
            accumulated_turns.append(turn)

    return output_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Reading input file: %s", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    log.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    all_output_rows = []
    skipped = 0
    total = len(df)

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        expanded = process_row(row_dict)
        if not expanded:
            skipped += 1
        all_output_rows.extend(expanded)

        if (idx + 1) % 2000 == 0:
            log.info(
                "Processed %d / %d original rows → %d expanded rows so far",
                idx + 1, total, len(all_output_rows),
            )

    log.info(
        "Finished. Skipped %d rows. Total expanded rows: %d",
        skipped, len(all_output_rows),
    )

    if not all_output_rows:
        log.error("No output rows generated.")
        sys.exit(1)

    output_df = pd.DataFrame(all_output_rows)
    for col in ("conversation_so_far", "tool_list", "selected_tools"):
        if col not in output_df.columns:
            output_df[col] = ""

    log.info("Writing output file: %s", OUTPUT_FILE)
    output_df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)
    log.info("Done → %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
