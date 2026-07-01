"""Download and combine Hugging Face datasets for the gated agentic CoreModel.

Pulls open datasets across the domains the model must cover — **finance /
accounting, data analytics (SQL), coding, reasoning, and tool/function
calling** — and normalizes every row into ONE of three schemas that the
existing training pipeline (`experiments/gating/data.py :: MixedModeDataset`)
already understands:

    reasoning : columns [prompt, context, thinking, answer]   -> THINK + RESPOND
    tools     : columns [query, answers, tools]               -> TOOL
    plain     : columns [prompts, answers]                    -> RESPOND

The three combined CSVs (``data/combined_reasoning.csv`` etc.) can therefore be
fed to training *altogether* — the latent gate sees reasoning traces, tool
calls, and direct answers in one mixed stream.

Design notes
------------
* Each source is declared once in ``REGISTRY`` with a ``mapper`` that converts a
  raw row into the target schema (or returns ``None`` to drop the row).
* Downloads are **streamed** and capped per dataset so you can build a workable
  mixture without pulling hundreds of GB. Use ``--max-per-dataset`` to scale.
* Any dataset that fails to load (gated / offline / renamed) is skipped with a
  warning — the run still produces the combined files from whatever succeeded.
* A ``manifest.json`` records how many rows each source contributed.

Usage
-----
    python -m scripts.download_datasets --max-per-dataset 5000
    python -m scripts.download_datasets --only gsm8k finance_alpaca --max-per-dataset 200
    python -m scripts.download_datasets --list          # show available sources
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

try:
    from datasets import load_dataset
except ImportError as e:  # pragma: no cover
    raise SystemExit("Please `pip install datasets` to run this script.") from e


# --------------------------------------------------------------------------- #
# Row mappers: raw HF row -> normalized dict for one of the three schemas.
# Return None to skip a row.
# --------------------------------------------------------------------------- #
def _clean(x) -> str:
    return "" if x is None else str(x).strip()


def map_gsm8k(row) -> Optional[Dict]:
    """GSM8K: split the CoT rationale from the final '#### <answer>'."""
    q, a = _clean(row.get("question")), _clean(row.get("answer"))
    if not q or "####" not in a:
        return None
    thinking, _, final = a.partition("####")
    return {"prompt": q, "context": "", "thinking": thinking.strip(), "answer": final.strip()}


def map_orca_math(row) -> Optional[Dict]:
    q, a = _clean(row.get("question")), _clean(row.get("answer"))
    if not q or not a:
        return None
    # No explicit trace separator; treat the worked solution as the reasoning
    # and the last line as the answer.
    lines = [ln for ln in a.splitlines() if ln.strip()]
    answer = lines[-1] if lines else a
    return {"prompt": q, "context": "", "thinking": a, "answer": answer}


def map_open_r1_math(row) -> Optional[Dict]:
    q = _clean(row.get("problem") or row.get("question"))
    sol = _clean(row.get("solution"))
    ans = _clean(row.get("answer"))
    if not q or not sol:
        return None
    return {"prompt": q, "context": "", "thinking": sol, "answer": ans or sol.splitlines()[-1]}


def map_alpaca_like(instr_key="instruction", input_key="input", out_key="output"):
    def _m(row) -> Optional[Dict]:
        instr = _clean(row.get(instr_key))
        ctx = _clean(row.get(input_key))
        out = _clean(row.get(out_key))
        if not instr or not out:
            return None
        prompt = instr if not ctx else f"{instr}\n\n{ctx}"
        return {"prompts": prompt, "answers": out}
    return _m


def map_finance_instruct(row) -> Optional[Dict]:
    # Sujet-Finance-Instruct-177k style: user_prompt / answer (+ system_prompt).
    q = _clean(row.get("user_prompt") or row.get("question") or row.get("instruction"))
    a = _clean(row.get("answer") or row.get("output"))
    if not q or not a:
        return None
    return {"prompts": q, "answers": a}


def map_text_to_sql(row) -> Optional[Dict]:
    # sql-create-context: question / context (schema) / answer (SQL).
    q = _clean(row.get("question") or row.get("sql_prompt"))
    schema = _clean(row.get("context") or row.get("sql_context"))
    sql = _clean(row.get("answer") or row.get("sql"))
    if not q or not sql:
        return None
    prompt = f"{q}\n\n### Database schema:\n{schema}" if schema else q
    return {"prompts": prompt, "answers": sql}


def map_xlam_tools(row) -> Optional[Dict]:
    query = _clean(row.get("query"))
    answers = row.get("answers")
    tools = row.get("tools")
    if not query or answers is None or tools is None:
        return None
    # Keep them JSON-serializable strings; the tool loader parses them.
    to_str = lambda v: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return {"query": query, "answers": to_str(answers), "tools": to_str(tools)}


# --------------------------------------------------------------------------- #
# Registry of sources.
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    name: str
    hf_id: str
    target: str  # "reasoning" | "tools" | "plain"
    mapper: Callable
    domain: str = ""
    config: Optional[str] = None
    split: str = "train"
    default_cap: int = 5000
    trust_remote_code: bool = False


REGISTRY: List[Source] = [
    # --- reasoning (with rationales) ---
    Source("gsm8k", "openai/gsm8k", "reasoning", map_gsm8k, domain="math-reasoning",
           config="main"),
    Source("orca_math", "microsoft/orca-math-word-problems-200k", "reasoning",
           map_orca_math, domain="math-reasoning"),
    Source("open_r1_math", "open-r1/OpenR1-Math-220k", "reasoning", map_open_r1_math,
           domain="reasoning", config="default"),

    # --- finance / accounting ---
    Source("finance_alpaca", "gbharti/finance-alpaca", "plain",
           map_alpaca_like("instruction", "input", "output"), domain="finance"),
    Source("sujet_finance", "sujet-ai/Sujet-Finance-Instruct-177k", "plain",
           map_finance_instruct, domain="finance-accounting"),

    # --- data analytics (text-to-SQL) ---
    Source("sql_create_context", "b-mc2/sql-create-context", "plain",
           map_text_to_sql, domain="data-analytics"),
    Source("synthetic_text_to_sql", "gretelai/synthetic_text_to_sql", "plain",
           map_text_to_sql, domain="data-analytics"),

    # --- coding ---
    Source("python_code_instructions", "iamtarun/python_code_instructions_18k_alpaca",
           "plain", map_alpaca_like("instruction", "input", "output"), domain="coding"),
    Source("code_alpaca", "sahil2801/CodeAlpaca-20k", "plain",
           map_alpaca_like("instruction", "input", "output"), domain="coding"),

    # --- tool / function calling ---
    Source("xlam_tools", "Salesforce/xlam-function-calling-60k", "tools",
           map_xlam_tools, domain="tool-calling"),
]

SCHEMA_COLUMNS = {
    "reasoning": ["prompt", "context", "thinking", "answer"],
    "tools": ["query", "answers", "tools"],
    "plain": ["prompts", "answers"],
}
OUTPUT_FILES = {
    "reasoning": "combined_reasoning.csv",
    "tools": "combined_tools.csv",
    "plain": "combined_plain.csv",
}


def collect_source(src: Source, cap: int, streaming: bool) -> List[Dict]:
    """Load and normalize up to ``cap`` rows from one source."""
    print(f"[{src.name}] loading {src.hf_id} (target={src.target}, domain={src.domain}) ...")
    kwargs = dict(split=src.split, streaming=streaming)
    if src.config:
        kwargs["name"] = src.config
    if src.trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        ds = load_dataset(src.hf_id, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[{src.name}] SKIPPED — could not load ({type(e).__name__}: {e})")
        return []

    rows: List[Dict] = []
    try:
        for i, raw in enumerate(ds):
            if len(rows) >= cap:
                break
            try:
                norm = src.mapper(raw)
            except Exception:  # noqa: BLE001
                norm = None
            if norm:
                norm["source"] = src.name
                norm["domain"] = src.domain
                rows.append(norm)
    except Exception as e:  # noqa: BLE001
        print(f"[{src.name}] iteration stopped early ({type(e).__name__}: {e})")

    print(f"[{src.name}] collected {len(rows)} rows")
    return rows


def dedup(rows: List[Dict], key_fields: List[str]) -> List[Dict]:
    seen, out = set(), []
    for r in rows:
        key = "␟".join(re.sub(r"\s+", " ", _clean(r.get(f))) for f in key_fields)
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--max-per-dataset", type=int, default=5000,
                        help="Cap rows collected per source (overrides per-source default).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Restrict to these source names (see --list).")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Download full datasets instead of streaming (uses more disk).")
    parser.add_argument("--list", action="store_true", help="List available sources and exit.")
    args = parser.parse_args()

    if args.list:
        print(f"{'name':<26}{'domain':<20}{'target':<10}hf_id")
        for s in REGISTRY:
            print(f"{s.name:<26}{s.domain:<20}{s.target:<10}{s.hf_id}")
        return

    sources = REGISTRY
    if args.only:
        wanted = set(args.only)
        sources = [s for s in REGISTRY if s.name in wanted]
        missing = wanted - {s.name for s in sources}
        if missing:
            print(f"[WARN] unknown sources ignored: {sorted(missing)}")

    buckets: Dict[str, List[Dict]] = {"reasoning": [], "tools": [], "plain": []}
    manifest = {"sources": {}, "totals": {}}

    for src in sources:
        cap = min(args.max_per_dataset, src.default_cap) if args.max_per_dataset else src.default_cap
        rows = collect_source(src, cap, streaming=not args.no_streaming)
        buckets[src.target].extend(rows)
        manifest["sources"][src.name] = {"rows": len(rows), "target": src.target, "domain": src.domain}

    os.makedirs(args.out_dir, exist_ok=True)
    dedup_keys = {"reasoning": ["prompt", "answer"], "tools": ["query"], "plain": ["prompts"]}

    for target, rows in buckets.items():
        rows = dedup(rows, dedup_keys[target])
        # Reorder to schema columns; keep source/domain as trailing metadata.
        cols = SCHEMA_COLUMNS[target] + ["source", "domain"]
        df = pd.DataFrame(rows)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols] if len(df) else pd.DataFrame(columns=cols)
        # Shuffle so domains interleave (deterministic).
        if len(df):
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        out_path = os.path.join(args.out_dir, OUTPUT_FILES[target])
        df.to_csv(out_path, index=False)
        manifest["totals"][target] = len(df)
        print(f"wrote {len(df):>7} rows -> {out_path}")

    manifest_path = os.path.join(args.out_dir, "combined_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {manifest_path}")
    print("\nPoint experiments/gating/config.py at these files (reasoning/tool/plain) to train on the mixture.")


if __name__ == "__main__":
    main()
