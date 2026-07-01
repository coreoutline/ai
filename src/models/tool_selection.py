"""Zero-shot tool-selection arm (the TOOL branch of the gated CoreModel).

Mirrors how ``facebook/bart-large-mnli`` does zero-shot classification: the user
prompt is the NLI *premise*, and each candidate tool is turned into a
*hypothesis* ("This request requires a tool that can {description}."). The
reconstructed BART-MNLI model scores each pair; the entailment probability is
the tool's relevance score.

Selection is **multi-label** (sigmoid-style): each tool's score is
``softmax([contradiction, entailment])[entailment]`` — an independent 0..1
probability, exactly like the HF pipeline's ``multi_label=True`` — so several
tools can be selected. Selected tools then **constrain generation**: their
signatures are rendered into the context before the CoreModel emits the actual
function-call arguments.

Typical flow (after the gate routes a position to TOOL)::

    selector = load_tool_selector("models/bart_mnli_tool_selector.pth")
    result = run_tool_arm(core_model, core_tokenizer, prompt, tools, selector)
    # result["selected"] -> [{name, score, ...}], result["generation"] -> str
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.models.bart_mnli import BartForSequenceClassification, BartMnliConfig

MNLI_NAME = "facebook/bart-large-mnli"
CONTRADICTION_ID = 0
ENTAILMENT_ID = 2

DEFAULT_HYPOTHESIS_TEMPLATE = "This request requires a tool that can {}."


@dataclass
class ToolScore:
    name: str
    score: float
    description: str = ""
    parameters: Any = None
    tool: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Tool parsing / rendering helpers.
# --------------------------------------------------------------------------- #
def parse_tools(tools: Any) -> List[Dict[str, Any]]:
    """Normalize a tools spec (JSON string or list) into a list of dicts."""
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return []
    if isinstance(tools, dict):
        tools = [tools]
    parsed = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or t.get("function", {}).get("name") or ""
        desc = t.get("description") or t.get("function", {}).get("description") or ""
        params = t.get("parameters") or t.get("function", {}).get("parameters") or {}
        parsed.append({"name": name, "description": desc, "parameters": params})
    return parsed


def _tool_label(tool: Dict[str, Any]) -> str:
    """Text used as the NLI hypothesis fill for a tool."""
    return (tool.get("description") or tool.get("name") or "").strip()


def _render_signature(tool: Dict[str, Any]) -> str:
    """Human-readable signature line, e.g. ``get_weather(location, date): ...``."""
    name = tool.get("name", "tool")
    params = tool.get("parameters") or {}
    arg_names: List[str] = []
    if isinstance(params, dict):
        # xLAM style {name: {type, description}} OR JSON-schema {properties: {...}}.
        props = params.get("properties", params)
        if isinstance(props, dict):
            arg_names = list(props.keys())
    elif isinstance(params, list):
        arg_names = [p.get("name", "") for p in params if isinstance(p, dict)]
    sig = f"{name}({', '.join(a for a in arg_names if a)})"
    desc = tool.get("description", "")
    return f"{sig}: {desc}".strip()


# --------------------------------------------------------------------------- #
# The selector.
# --------------------------------------------------------------------------- #
class ZeroShotToolSelector:
    """Score candidate tools against a prompt via reconstructed BART-MNLI."""

    def __init__(
        self,
        model: BartForSequenceClassification,
        tokenizer,
        hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.hypothesis_template = hypothesis_template
        self.max_length = max_length

    @torch.no_grad()
    def score(self, prompt: str, tools: Any) -> List[ToolScore]:
        """Return an entailment score (0..1) for every candidate tool."""
        parsed = parse_tools(tools)
        if not parsed:
            return []

        hypotheses = [self.hypothesis_template.format(_tool_label(t)) for t in parsed]
        premises = [prompt] * len(hypotheses)
        enc = self.tokenizer(
            premises, hypotheses,
            return_tensors="pt", padding=True, truncation=True, max_length=self.max_length,
        ).to(self.device)

        logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        # multi_label: entailment vs contradiction per tool -> independent prob.
        entail_contra = logits[:, [CONTRADICTION_ID, ENTAILMENT_ID]]
        probs = F.softmax(entail_contra, dim=-1)[:, 1]

        scores = [
            ToolScore(
                name=t["name"], score=float(p), description=t.get("description", ""),
                parameters=t.get("parameters"), tool=t,
            )
            for t, p in zip(parsed, probs.tolist())
        ]
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores

    def select(
        self, prompt: str, tools: Any, threshold: float = 0.5, top_k: Optional[int] = None
    ) -> List[ToolScore]:
        """Select tools whose entailment score exceeds ``threshold``.

        If none clear the threshold, fall back to the single best tool so the
        TOOL branch always produces at least one candidate. ``top_k`` caps the
        number returned.
        """
        scores = self.score(prompt, tools)
        if not scores:
            return []
        selected = [s for s in scores if s.score >= threshold]
        if not selected:
            selected = scores[:1]
        if top_k is not None:
            selected = selected[:top_k]
        return selected

    def render_constraint(self, selected: List[ToolScore]) -> str:
        """Render the selected tools' signatures for context injection."""
        lines = ["[Selected tools]"]
        for s in selected:
            lines.append(f"- {_render_signature(s.tool)}  (relevance={s.score:.2f})")
        return "\n".join(lines)

    def build_tool_prompt(self, prompt: str, selected: List[ToolScore]) -> str:
        """Prompt that constrains the model to emit calls for the selected tools."""
        constraint = self.render_constraint(selected)
        return (
            f"{prompt.rstrip()}\n\n{constraint}\n\n"
            "Emit the function call(s) as JSON for the selected tool(s).\n"
            "### Function Calls:\n"
        )


def load_tool_selector(
    weights_path: str = "models/bart_mnli_tool_selector.pth",
    device: Optional[str] = None,
    hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE,
) -> ZeroShotToolSelector:
    """Load the reconstructed BART-MNLI selector + its tokenizer."""
    from transformers import AutoTokenizer

    model = BartForSequenceClassification(BartMnliConfig())
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(MNLI_NAME)
    return ZeroShotToolSelector(model, tokenizer, hypothesis_template=hypothesis_template, device=device)


# --------------------------------------------------------------------------- #
# Integration with the gated CoreModel.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def detect_tool_gate(core_model, core_tokenizer, prompt: str, device: Optional[str] = None) -> Dict[str, Any]:
    """Run CoreModel on the prompt and read the gate's mode at the last position."""
    from src.core.gating import Mode

    device = device or next(core_model.parameters()).device
    enc = core_tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    out = core_model(input_ids=input_ids, return_gates=True)
    gw = out["gate_weights"]
    if gw is None:
        return {"mode": None, "is_tool": False, "mode_probs": None}
    last = gw.mean(dim=0)[:, -1, :]  # avg over gate layers -> [B, num_modes]
    probs = last[0]
    mode_id = int(probs.argmax())
    return {
        "mode": Mode(mode_id).name,
        "mode_id": mode_id,
        "is_tool": mode_id == int(Mode.TOOL),
        "mode_probs": probs.tolist(),
    }


@torch.no_grad()
def run_tool_arm(
    core_model,
    core_tokenizer,
    prompt: str,
    tools: Any,
    selector: ZeroShotToolSelector,
    *,
    gate_check: bool = True,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    max_new_tokens: int = 128,
    generate: bool = True,
) -> Dict[str, Any]:
    """Full TOOL branch: (optionally) confirm the gate, select tools, generate.

    Returns a dict with the detected gate mode, ranked tool scores, the selected
    tools, the constrained prompt, and (optionally) the generated call text.
    """
    result: Dict[str, Any] = {}

    if gate_check:
        gate = detect_tool_gate(core_model, core_tokenizer, prompt)
        result["gate"] = gate

    scores = selector.score(prompt, tools)
    selected = selector.select(prompt, tools, threshold=threshold, top_k=top_k)
    result["scores"] = [vars(s) for s in scores]
    result["selected"] = [{"name": s.name, "score": s.score} for s in selected]

    constrained_prompt = selector.build_tool_prompt(prompt, selected)
    result["constrained_prompt"] = constrained_prompt

    if generate and selected:
        device = next(core_model.parameters()).device
        enc = core_tokenizer(constrained_prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        gen_ids = core_model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            eos_id=getattr(core_model.config, "eos_token_id", None),
        )
        new_tokens = gen_ids[0, input_ids.shape[1]:]
        result["generation"] = core_tokenizer.decode(new_tokens, skip_special_tokens=True)

    return result
