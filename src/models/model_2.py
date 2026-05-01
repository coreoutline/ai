import pandas as pd
import torch
from torch import nn
from typing import Optional, Tuple
import math

# Import everything from the new core modules to maintain backwards compatibility
from src.core import (
    CoreOutlineRotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    CoreOutlineMLP,
    CoreOutlineAttention,
    CoreOutlineDecoderLayer,
    CoreOutlineModel,
    CoreOutlineForCausalLM,
    CoreOutlineConfig,
    create_coreoutline_qwen_model
)

config = CoreOutlineConfig(
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=2816,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=16,
    max_position_embeddings=32768,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
)
# config = CoreOutlineConfig(
#     vocab_size=151936,
#     hidden_size=1536,
#     intermediate_size=8960,
#     num_hidden_layers=28,
#     num_attention_heads=24,   # 1536 / 64
#     num_key_value_heads=4,    # 4 * 64 = 256 for k,v proj
#     max_position_embeddings=32768,
#     initializer_range=0.02,
#     rms_norm_eps=1e-6,
#     rope_theta=1000000.0,
# )


# Create the model
core_model = create_coreoutline_qwen_model(config)

# print(core_model)
