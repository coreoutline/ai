"""Extract the tool list (JSON) embedded in the system prompt of each conversation.

Reads data/tool-use-multiturn-reasoning.csv, parses the `conversations` column
(a stringified list of {"from", "value"} turns), pulls the JSON tool
definitions out of the <tools>...</tools> block inside the first (system)
turn, and writes the result back out as a new `tool_list` column.

Why the custom parsing instead of json.loads(conversations) directly:
the `conversations` field is a Python repr() of a list of dicts, not JSON
(single-quoted strings, embedded escapes), AND the CSV export dropped the
commas between top-level list elements -- consecutive dict entries are
separated only by a real newline. We restore the missing commas before
handing the text to ast.literal_eval, which correctly reverses the
repr()-style escaping (unlike a manual regex unescape, which breaks on the
mixed ' / " quoting Python's repr() chooses per-string).

The <tools> block itself is usually valid JSON (double-quoted), but a
subset of rows (source == "Nous-Hermes") contain a Python dict repr
instead (single-quoted). We try json.loads first and fall back to
ast.literal_eval so both formats normalize to the same JSON output.

Every extracted tool is further normalized to a single common shape,
since the three source datasets (ToolAce, Glaive, Nous-Hermes) disagree
on both the envelope and the JSON Schema type names:
  - ToolAce ships flat tools ({"name", "description", "parameters",
    "required": null}) instead of the OpenAI function-calling envelope
    ({"type": "function", "function": {...}}) used by Glaive/Nous-Hermes.
    We wrap ToolAce tools into that envelope and drop the always-null
    top-level "required" key.
  - ToolAce also uses Python type names ("dict", "int", "float") inside
    "parameters" instead of JSON Schema names ("object", "integer",
    "number"). We rewrite those recursively wherever a "type" key
    appears in a tool's parameter schema.
"""

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

MISSING_COMMA_RE = re.compile(r"\}\n")
TOOLS_BLOCK_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)

JSON_SCHEMA_TYPE_MAP = {
    "dict": "object",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "str": "string",
}


def normalize_schema_types(node):
    """Recursively rewrite Python-style type names to JSON Schema names."""
    if isinstance(node, dict):
        normalized = {}
        for key, value in node.items():
            if key == "type" and isinstance(value, str):
                normalized[key] = JSON_SCHEMA_TYPE_MAP.get(value, value)
            else:
                normalized[key] = normalize_schema_types(value)
        return normalized
    if isinstance(node, list):
        return [normalize_schema_types(item) for item in node]
    return node


def normalize_tool(tool: dict) -> dict:
    """Normalize a single tool definition to the OpenAI function-calling shape:
    {"type": "function", "function": {"name", "description", "parameters"}}.
    """
    if tool.get("type") == "function" and "function" in tool:
        function = tool["function"]
    else:
        # Flat (ToolAce) shape: {"name", "description", "parameters", "required": null}.
        function = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        }
    function = normalize_schema_types(function)
    return {"type": "function", "function": function}


def extract_system_prompt(conversation_str: str) -> str:
    """Parse the `conversations` cell and return the system turn's text."""
    fixed = MISSING_COMMA_RE.sub("},\n", conversation_str)
    turns = ast.literal_eval(fixed)
    for turn in turns:
        if turn.get("from") == "system":
            return turn["value"]
    raise ValueError("no system turn found")


def extract_tool_list(system_prompt: str) -> list:
    """Pull the JSON tool definitions out of the <tools>...</tools> block."""
    matches = TOOLS_BLOCK_RE.findall(system_prompt)
    if not matches:
        raise ValueError("no <tools> block found")
    # The prompt boilerplate mentions an empty "<tools> </tools>" tag before
    # the real one, so the block with the actual definitions is the last match.
    content = matches[-1]
    try:
        tools = json.loads(content)
    except json.JSONDecodeError:
        tools = ast.literal_eval(content)
    return [normalize_tool(tool) for tool in tools]


def process(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)

    tool_lists = []
    failures = []
    for i, conversation_str in enumerate(df["conversations"]):
        try:
            system_prompt = extract_system_prompt(conversation_str)
            tools = extract_tool_list(system_prompt)
            tool_lists.append(json.dumps(tools))
        except Exception as exc:
            failures.append((i, str(exc)))
            tool_lists.append(None)

    df["tool_list"] = tool_lists

    if failures:
        print(f"WARNING: failed to extract tool_list for {len(failures)} row(s):")
        for idx, err in failures[:10]:
            print(f"  row {idx}: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
    else:
        print(f"Extracted tool_list for all {len(df)} rows.")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/tool-use-multiturn-reasoning.csv"),
        help="Path to the source CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tool-use-multiturn-reasoning-with-tool-list.csv"),
        help="Path to write the CSV with the added tool_list column.",
    )
    args = parser.parse_args()
    process(args.input, args.output)


if __name__ == "__main__":
    main()
