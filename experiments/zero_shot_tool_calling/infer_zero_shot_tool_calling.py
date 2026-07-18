"""
Inference script for the zero-shot tool-selection classifier (reconstructed
BART-large-MNLI), trained by train_zero_shot_tool_calling.py in this
directory.

Scores every candidate tool against a prompt as an NLI (premise, hypothesis)
pair and reports the entailment probability -- see
src/models/tool_selection.py for the ZeroShotToolSelector this script wraps.

Examples:
    # Inline tools JSON
    python3.11 experiments/zero_shot_tool_calling/infer_zero_shot_tool_calling.py \\
        --checkpoint ./models/bart_mnli_tool_selector.pth \\
        --prompt "What is the weather in Tokyo tomorrow?" \\
        --tools '[{"name": "get_weather", "description": "Get the weather forecast for a location."}, \\
                  {"name": "create_doc", "description": "Create a new Google Doc."}]'

    # Tools from a file (list of {name, description[, parameters]} or xLAM-style
    # {"function": {name, description, parameters}} dicts)
    python3.11 experiments/zero_shot_tool_calling/infer_zero_shot_tool_calling.py \\
        --checkpoint ./models/bart_mnli_tool_selector.pth \\
        --prompt "Book me a flight to Paris next Friday." \\
        --tools-file ./data/sample_tools.json \\
        --threshold 0.5 \\
        --top-k 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.tool_selection import load_tool_selector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score/select tools for a prompt with the zero-shot tool selector"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./models/bart_mnli_tool_selector.pth",
        help="Path to the trained BART-MNLI tool-selector checkpoint",
    )
    parser.add_argument("--prompt", type=str, help="User prompt / conversation-so-far text")
    parser.add_argument(
        "--prompt-file", type=str, default=None,
        help="Read the prompt from a text file instead of --prompt",
    )

    tools_group = parser.add_mutually_exclusive_group(required=True)
    tools_group.add_argument("--tools", type=str, help="Inline JSON list of candidate tools")
    tools_group.add_argument("--tools-file", type=str, help="Path to a JSON file listing candidate tools")

    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Entailment-probability threshold for selecting a tool (default: 0.5)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Cap the number of selected tools (applied after --threshold)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto-detect)")
    return parser.parse_args()


def load_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt
    raise ValueError("Provide either --prompt or --prompt-file")


def load_tools(args):
    if args.tools_file:
        return json.loads(Path(args.tools_file).read_text(encoding="utf-8"))
    return json.loads(args.tools)


def main():
    args = parse_args()
    prompt = load_prompt(args)
    tools = load_tools(args)

    selector = load_tool_selector(weights_path=args.checkpoint, device=args.device)

    scores = selector.score(prompt, tools)
    print(f"Prompt: {prompt}\n")
    print("Ranked tool scores:")
    for s in scores:
        print(f"  {s.score:.3f}  {s.name}")

    selected = selector.select(prompt, tools, threshold=args.threshold, top_k=args.top_k)
    print(f"\nSelected tools (threshold={args.threshold}, top_k={args.top_k}):")
    if not selected:
        print("  (none)")
    else:
        for s in selected:
            print(f"  {s.score:.3f}  {s.name}")
        print()
        print(selector.render_constraint(selected))


if __name__ == "__main__":
    main()
