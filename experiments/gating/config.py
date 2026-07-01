"""Experiment configuration for the gated CoreModel.

A deliberately small model so the experiment is runnable on a single GPU (or CPU
for the smoke test). All gating hyperparameters live on :class:`CoreModelConfig`.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# Segment ids used only for *evaluation* (mode-alignment probe). They mirror
# src.core.gating.Mode so the confusion matrix lines up with the router modes.
SEGMENT_IGNORE = -100
SEG_THINK = 0
SEG_TOOL = 1
SEG_RESPOND = 2
SEG_DONE = 3
SEGMENT_NAMES = ["THINK", "TOOL", "RESPOND", "DONE"]


@dataclass
class ExperimentConfig:
    # --- data ---
    tokenizer_name: str = "Qwen/Qwen1.5-0.5B"
    reasoning_csv: str = "data/reasoning_smoke_test.csv"
    tool_csv: str = "data/xlam_function_calling_smoke_test.csv"
    plain_csv: Optional[str] = "data/nyx-finance-instruct.csv"
    # Optional pretrained init (transferred from nyx_reasoning). strict=False so
    # only the backbone/head load and the gate stays freshly initialized.
    init_weights_path: Optional[str] = None
    max_length: int = 1024
    max_samples_per_source: Optional[int] = None  # cap each source; None = all
    val_fraction: float = 0.1
    test_fraction: float = 0.05

    # --- model (small, feasible) ---
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    max_position_embeddings: int = 1024

    # --- gating ---
    gate_layer_indices: List[int] = field(default_factory=lambda: [4])
    share_gate: bool = True
    router_aux_loss_coef: float = 1e-2
    router_z_loss_coef: float = 1e-3
    gate_noise_std: float = 0.3
    gate_temperature: float = 1.0
    use_straight_through: bool = False
    done_threshold: float = 0.5
    # Load-balance target prior [THINK, TOOL, RESPOND, DONE]. DONE kept small so
    # it fires ~once per sequence instead of ~25% of tokens.
    mode_balance_target: List[float] = field(
        default_factory=lambda: [0.33, 0.30, 0.33, 0.04]
    )

    # --- optimization ---
    batch_size: int = 4
    num_epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    # Ramp the aux-loss coefficient 0 -> 1x over this many steps so the LM
    # stabilizes before routing pressure is applied.
    aux_ramp_steps: int = 500
    eval_every: int = 200
    seed: int = 42

    # --- io ---
    checkpoint_path: str = "experiments/gating/checkpoints/core_model.pt"
    log_path: str = "experiments/gating/logs/train_log.jsonl"


# Output files produced by scripts/download_datasets.py.
COMBINED_REASONING_CSV = "data/combined_reasoning.csv"
COMBINED_TOOLS_CSV = "data/combined_tools.csv"
COMBINED_PLAIN_CSV = "data/combined_plain.csv"


def use_combined_datasets(exp: "ExperimentConfig") -> "ExperimentConfig":
    """Point an ExperimentConfig at the combined HF datasets."""
    exp.reasoning_csv = COMBINED_REASONING_CSV
    exp.tool_csv = COMBINED_TOOLS_CSV
    exp.plain_csv = COMBINED_PLAIN_CSV
    return exp


def use_nyx_architecture(exp: "ExperimentConfig") -> "ExperimentConfig":
    """Match the pretrained nyx_reasoning backbone so its weights can be loaded.

    (hidden=1024, 24 layers, 16 heads, intermediate=2816). Gate sits mid-stack.
    """
    exp.hidden_size = 1024
    exp.intermediate_size = 2816
    exp.num_hidden_layers = 24
    exp.num_attention_heads = 16
    exp.num_key_value_heads = 16
    exp.max_position_embeddings = 32768
    exp.gate_layer_indices = [12]
    return exp


def build_model_config(exp: ExperimentConfig):
    """Translate an :class:`ExperimentConfig` into a ``CoreModelConfig``."""
    from src.models.core_model import CoreModelConfig
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(exp.tokenizer_name)
    vocab_size = tok.vocab_size
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    return CoreModelConfig(
        vocab_size=vocab_size,
        hidden_size=exp.hidden_size,
        intermediate_size=exp.intermediate_size,
        num_hidden_layers=exp.num_hidden_layers,
        num_attention_heads=exp.num_attention_heads,
        num_key_value_heads=exp.num_key_value_heads,
        max_position_embeddings=exp.max_position_embeddings,
        pad_token_id=pad_id,
        eos_token_id=tok.eos_token_id,
        gate_layer_indices=exp.gate_layer_indices,
        share_gate=exp.share_gate,
        router_aux_loss_coef=exp.router_aux_loss_coef,
        router_z_loss_coef=exp.router_z_loss_coef,
        gate_noise_std=exp.gate_noise_std,
        gate_temperature=exp.gate_temperature,
        use_straight_through=exp.use_straight_through,
        done_threshold=exp.done_threshold,
        mode_balance_target=exp.mode_balance_target,
    )
