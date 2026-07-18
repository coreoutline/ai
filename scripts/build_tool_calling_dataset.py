"""Download several tool-calling datasets from Hugging Face and merge them into one CSV.

Normalizes every source into the same three-column schema already consumed by
``src/training/tool_calling_data.py`` and ``experiments/gating/data.py``:

    query   : the user request (str)
    tools   : the available function/tool definitions (JSON string)
    answers : the expected function call(s) (JSON string, list of {name, arguments})

Sources (see REGISTRY below):
    xlam              Salesforce/xlam-function-calling-60k   -- already query/tools/answers
    glaive_v2         glaiveai/glaive-function-calling-v2    -- system/chat text blob, parsed
    hermes_singleturn NousResearch/hermes-function-calling-v1 (config=func_calling_singleturn)
                                                              -- ShareGPT-style conversations

Design notes
------------
* Each source has a ``mapper`` that converts one raw HF row into the target
  schema, or returns ``None`` to drop the row (e.g. glaive rows with no
  function call, or rows missing a query/tools/answer).
* glaive_v2 and hermes embed JSON *inside* free text (``<functioncall> {...}``,
  ``<tool_call>{...}</tool_call>``). Regex with a non-greedy ``.*?`` truncates
  at the first ``}``, which breaks on nested objects (e.g. an ``arguments``
  value that is itself a JSON string). ``_find_balanced_json_object`` instead
  scans forward tracking brace depth and quote state, so nested braces and
  braces inside quoted strings don't end the match early.
* Salesforce/xlam-function-calling-60k is a gated dataset; if
  ``HUGGINGFACE_API_KEY`` (or ``HF_TOKEN``) is set in the environment/.env,
  the script logs in to huggingface_hub before downloading so it can be
  pulled non-interactively.
* Downloads are streamed and capped per source (``--max-per-dataset``) so a
  full run doesn't require pulling all of xlam's 60k / glaive's 113k rows.
* Any source that fails to load (gated without a token, offline, renamed) is
  skipped with a warning -- the run still produces a CSV from whatever
  succeeded, and a manifest.json records how many rows each source contributed.

Usage
-----
    python -m scripts.build_tool_calling_dataset
    python -m scripts.build_tool_calling_dataset --max-per-dataset 2000
    python -m scripts.build_tool_calling_dataset --only xlam glaive_v2
    python -m scripts.build_tool_calling_dataset --list
"""

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    from datasets import load_dataset
except ImportError as e:  # pragma: no cover
    raise SystemExit("Please `pip install datasets` to run this script.") from e


# --------------------------------------------------------------------------- #
# Small parsing helpers shared by the mappers.
# --------------------------------------------------------------------------- #
def _clean(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _parse_json_like(text: str) -> Any:
    """Parse JSON, falling back to Python literal syntax (single-quoted dicts)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _find_balanced_json_object(text: str, from_idx: int = 0) -> Optional[Tuple[str, int]]:
    """Return (substring, end_index) of the first brace-balanced {...} at/after from_idx.

    Tracks quote state so braces inside quoted strings (single or double)
    don't affect the depth count -- unlike a regex ``\\{.*?\\}``, this
    correctly spans nested objects such as {"arguments": '{"a": 1}'}.
    """
    depth = 0
    start = None
    in_str = False
    str_char = ""
    escape = False
    for i in range(from_idx, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == str_char:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            str_char = ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1], i + 1
    return None


def _split_concatenated_json_objects(text: str) -> List[Any]:
    """Glaive's system prompt lists functions as back-to-back JSON objects (no array)."""
    objs, idx = [], 0
    while True:
        found = _find_balanced_json_object(text, idx)
        if found is None:
            break
        snippet, idx = found
        try:
            objs.append(_parse_json_like(snippet))
        except Exception:  # noqa: BLE001
            pass
    return objs


def _extract_json_objects_after_marker(text: str, marker: str) -> List[Any]:
    """Find every JSON object that starts just after each occurrence of `marker`."""
    objs, idx = [], 0
    while True:
        pos = text.find(marker, idx)
        if pos == -1:
            break
        found = _find_balanced_json_object(text, pos + len(marker))
        if found is None:
            break
        snippet, idx = found
        try:
            objs.append(_parse_json_like(snippet))
        except Exception:  # noqa: BLE001
            continue
    return objs


_GLAIVE_USER_RE = re.compile(r"USER:\s*(.*?)\s*ASSISTANT:", re.DOTALL)
_GLAIVE_TOOLS_MARKER = "Use them if required -"


# --------------------------------------------------------------------------- #
# Row mappers: raw HF row -> {"query", "tools", "answers"} or None to skip.
# --------------------------------------------------------------------------- #
def map_xlam(row: Dict) -> Optional[Dict]:
    query = _clean(row.get("query"))
    tools, answers = row.get("tools"), row.get("answers")
    if not query or tools is None or answers is None:
        return None
    return {"query": query, "tools": _stringify(tools), "answers": _stringify(answers)}


def map_glaive(row: Dict) -> Optional[Dict]:
    system, chat = _clean(row.get("system")), _clean(row.get("chat"))
    if not system or not chat:
        return None

    tools_text = system.split(_GLAIVE_TOOLS_MARKER, 1)[-1]
    tools = _split_concatenated_json_objects(tools_text)
    if not tools:
        return None

    user_match = _GLAIVE_USER_RE.search(chat)
    query = user_match.group(1).strip() if user_match else ""

    calls = []
    for call in _extract_json_objects_after_marker(chat, "<functioncall>"):
        if not isinstance(call, dict) or not call.get("name"):
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass  # leave as raw string if it isn't valid JSON
        calls.append({"name": call["name"], "arguments": args})

    if not query or not calls:
        return None
    return {"query": query, "tools": _stringify(tools), "answers": _stringify(calls)}


def map_hermes_singleturn(row: Dict) -> Optional[Dict]:
    tools_raw, conversations = row.get("tools"), row.get("conversations")
    if not tools_raw or not conversations:
        return None
    try:
        tools = _parse_json_like(tools_raw) if isinstance(tools_raw, str) else tools_raw
    except Exception:  # noqa: BLE001
        return None

    query = ""
    calls: List[Dict] = []
    for turn in conversations:
        role, value = turn.get("from"), _clean(turn.get("value"))
        if role == "human" and not query:
            query = value
        elif role == "gpt":
            calls.extend(
                c for c in _extract_json_objects_after_marker(value, "<tool_call>")
                if isinstance(c, dict) and c.get("name")
            )

    if not query or not calls:
        return None
    return {"query": query, "tools": _stringify(tools), "answers": _stringify(calls)}


# --------------------------------------------------------------------------- #
# Registry of sources.
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    name: str
    hf_id: str
    mapper: Callable[[Dict], Optional[Dict]]
    config: Optional[str] = None
    split: str = "train"
    default_cap: int = 20000


REGISTRY: List[Source] = [
    Source("xlam", "Salesforce/xlam-function-calling-60k", map_xlam, default_cap=20000),
    Source("glaive_v2", "glaiveai/glaive-function-calling-v2", map_glaive, default_cap=20000),
    Source("hermes_singleturn", "NousResearch/hermes-function-calling-v1", map_hermes_singleturn,
           config="func_calling_singleturn", default_cap=5000),
]

SCHEMA_COLUMNS = ["query", "tools", "answers", "source"]


def _maybe_login_to_hf() -> None:
    """Log in to huggingface_hub if a token is available, for gated datasets like xlam."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Hugging Face login failed ({type(e).__name__}: {e}); gated sources may fail.")


def collect_source(src: Source, cap: int, streaming: bool) -> List[Dict]:
    """Load and normalize up to `cap` rows from one source."""
    print(f"[{src.name}] loading {src.hf_id} (config={src.config}) ...")
    kwargs = dict(split=src.split, streaming=streaming)
    if src.config:
        kwargs["name"] = src.config
    try:
        ds = load_dataset(src.hf_id, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[{src.name}] SKIPPED -- could not load ({type(e).__name__}: {e})")
        return []

    rows: List[Dict] = []
    try:
        for raw in ds:
            if len(rows) >= cap:
                break
            try:
                norm = src.mapper(raw)
            except Exception:  # noqa: BLE001
                norm = None
            if norm:
                norm["source"] = src.name
                rows.append(norm)
    except Exception as e:  # noqa: BLE001
        print(f"[{src.name}] iteration stopped early ({type(e).__name__}: {e})")

    print(f"[{src.name}] collected {len(rows)} rows")
    return rows


def dedup_by_query(rows: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in rows:
        key = re.sub(r"\s+", " ", _clean(r.get("query"))).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/combined_tool_calling.csv", help="Output CSV path.")
    parser.add_argument("--max-per-dataset", type=int, default=None,
                        help="Cap rows collected per source (overrides per-source default).")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to these source names (see --list).")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Download full datasets instead of streaming (uses more disk).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the combined output.")
    parser.add_argument("--list", action="store_true", help="List available sources and exit.")
    args = parser.parse_args()

    if args.list:
        print(f"{'name':<20}{'default_cap':<14}hf_id")
        for s in REGISTRY:
            print(f"{s.name:<20}{s.default_cap:<14}{s.hf_id}{f' [{s.config}]' if s.config else ''}")
        return

    sources = REGISTRY
    if args.only:
        wanted = set(args.only)
        sources = [s for s in REGISTRY if s.name in wanted]
        missing = wanted - {s.name for s in sources}
        if missing:
            print(f"[WARN] unknown sources ignored: {sorted(missing)}")

    _maybe_login_to_hf()

    all_rows: List[Dict] = []
    manifest = {"sources": {}, "total_before_dedup": 0, "total_after_dedup": 0}

    for src in sources:
        cap = args.max_per_dataset if args.max_per_dataset else src.default_cap
        rows = collect_source(src, cap, streaming=not args.no_streaming)
        all_rows.extend(rows)
        manifest["sources"][src.name] = {"hf_id": src.hf_id, "rows": len(rows)}

    manifest["total_before_dedup"] = len(all_rows)
    all_rows = dedup_by_query(all_rows)
    manifest["total_after_dedup"] = len(all_rows)

    df = pd.DataFrame(all_rows, columns=SCHEMA_COLUMNS)
    if len(df):
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df):>7} rows -> {args.out}")

    manifest_path = os.path.splitext(args.out)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
