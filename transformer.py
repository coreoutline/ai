from attention_transformer import MultiHeadAttention
from feed_forward import FeedForward
from layer_normalization import LayerNormalization
import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in= cfg["emb_dim"], d_out = cfg["emb_dim"],
            context_length = cfg["context_length"], 
            num_heads=cfg["n_heads"], 
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNormalization(cfg["emb_dim"])
        self.norm2 = LayerNormalization(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        

    def forward(self, X):
        shortcut = X
        X = self.norm1(X)
        X = self.attn(X)
        X = self.drop_shortcut(X)
        X = X + shortcut

        shortcut = X
        X = self.norm2(X)
        X = self.ff(X)
        X = self.drop_shortcut(X)
        X = X + shortcut

        return X