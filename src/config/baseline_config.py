# Baseline Transformer Configuration

CORE_TRANSFORMER_CONFIG = {
    "vocab_size": 50257, # Default for gpt2 tokenizer
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12, 
    "drop_rate": 0.1,
    "qkv_bias": False,
    "lr": 1e-4
}
