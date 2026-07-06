"""Configuration for the multi-turn reasoning + tool-use experiment."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MultiTurnConfig:
    # --- data ---
    tokenizer_name: str = "Qwen/Qwen1.5-0.5B"
    csv_path: str = "data/tool-use-multiturn-reasoning.csv"
    max_length: int = 2048
    max_samples: Optional[int] = None
    val_fraction: float = 0.1
    test_fraction: float = 0.05

    # --- model (small default; use `nyx=True` to match pretrained backbone) ---
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    max_position_embeddings: int = 2048

    # --- gating (4 modes: THINK / TOOL / RESPOND / DONE) ---
    gate_layer_indices: List[int] = field(default_factory=lambda: [4])
    share_gate: bool = True
    router_aux_loss_coef: float = 1e-2
    router_z_loss_coef: float = 1e-3
    gate_noise_std: float = 0.3
    done_threshold: float = 0.5
    mode_balance_target: List[float] = field(default_factory=lambda: [0.33, 0.30, 0.33, 0.04])

    # --- optimization ---
    batch_size: int = 2
    num_epochs: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    aux_ramp_steps: int = 500
    eval_every: int = 200
    seed: int = 42

    # --- vocab (None = derive from tokenizer; nyx pins it to the checkpoint) ---
    vocab_size: Optional[int] = None

    # --- warm start ---
    init_weights_path: Optional[str] = None

    # --- io ---
    checkpoint_path: str = "experiments/multiturn/checkpoints/core_model_multiturn.pt"
    log_path: str = "experiments/multiturn/logs/train_log.jsonl"


def use_nyx_architecture(cfg: MultiTurnConfig) -> MultiTurnConfig:
    """Match the pretrained nyx_reasoning backbone (1024/24/16)."""
    cfg.hidden_size = 1024
    cfg.intermediate_size = 2816
    cfg.num_hidden_layers = 24
    cfg.num_attention_heads = 16
    cfg.num_key_value_heads = 16
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, 2048)
    cfg.gate_layer_indices = [12]
    # nyx_reasoning was trained with a padded embedding of 151936 rows; the
    # embedding must match the checkpoint exactly for warm-start to load.
    cfg.vocab_size = 151936
    return cfg


def resolve_vocab_size(cfg: MultiTurnConfig, tokenizer) -> int:
    """Safe vocab size: honor an explicit cfg value, else cover every token id.

    Qwen's ``tokenizer.vocab_size`` excludes added special tokens, so the real
    pad/eos ids can exceed it — the embedding must be large enough for all ids.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id or 0
    if cfg.vocab_size is not None:
        return cfg.vocab_size
    return max(len(tokenizer), tokenizer.vocab_size, pad_id + 1, eos_id + 1)


def build_model_config(cfg: MultiTurnConfig, vocab_size: int, pad_id: int, eos_id: int):
    from src.models.core_model import CoreModelConfig

    return CoreModelConfig(
        vocab_size=vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        gate_layer_indices=cfg.gate_layer_indices,
        share_gate=cfg.share_gate,
        router_aux_loss_coef=cfg.router_aux_loss_coef,
        router_z_loss_coef=cfg.router_z_loss_coef,
        gate_noise_std=cfg.gate_noise_std,
        done_threshold=cfg.done_threshold,
        mode_balance_target=cfg.mode_balance_target,
    )
