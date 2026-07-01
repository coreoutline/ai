"""From-scratch PyTorch reconstruction of ``facebook/bart-large-mnli``.

BART is a seq2seq (encoder-decoder) transformer. The MNLI variant adds a
``BartClassificationHead`` on top of the decoder that reads the representation of
the final ``</s>`` (EOS) token and predicts 3 NLI labels:

    0 = contradiction, 1 = neutral, 2 = entailment

This module reproduces the architecture exactly — same submodule names and
tensor shapes as the HF checkpoint — so weights transfer 1:1 (see
``scripts/transfer_bart_mnli.py``). It powers the zero-shot **tool-selection
arm**: score a prompt (premise) against candidate tools (hypotheses) and read
the entailment probability.

Reference: Lewis et al. 2019, *BART*. Architecture facts for bart-large:
d_model=1024, 12 enc + 12 dec layers, 16 heads, ffn=4096, gelu, post-norm
(normalize_before=False), no final layer norm, learned positions (offset 2),
tied token embeddings.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BartMnliConfig:
    vocab_size: int = 50265
    d_model: int = 1024
    encoder_layers: int = 12
    decoder_layers: int = 12
    encoder_attention_heads: int = 16
    decoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    decoder_ffn_dim: int = 4096
    max_position_embeddings: int = 1024
    num_labels: int = 3
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    decoder_start_token_id: int = 2
    scale_embedding: bool = False
    classifier_dropout: float = 0.0
    dropout: float = 0.0  # eval-time; kept 0 so reconstruction matches HF exactly


# Learned positions in BART are offset by 2 (ids 0/1 reserved).
POSITION_OFFSET = 2


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """Shift input ids one token to the right, prepending the decoder start id."""
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int) -> torch.Tensor:
    """[bsz, src_len] 1/0 mask -> additive [bsz, 1, tgt_len, src_len] mask."""
    bsz, src_len = mask.shape
    expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted = 1.0 - expanded
    return inverted.masked_fill(inverted.bool(), torch.finfo(dtype).min)


def _causal_mask(tgt_len: int, dtype: torch.dtype, device) -> torch.Tensor:
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    cond = torch.arange(tgt_len, device=device)
    mask.masked_fill_(cond < (cond + 1).view(tgt_len, 1), 0.0)
    return mask.to(dtype)[None, None, :, :]


class BartLearnedPositionalEmbedding(nn.Embedding):
    """Learned absolute positions with BART's +2 index offset."""

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__(num_positions + POSITION_OFFSET, embedding_dim)

    def forward(self, seq_len: int, device) -> torch.Tensor:
        positions = torch.arange(seq_len, dtype=torch.long, device=device) + POSITION_OFFSET
        return super().forward(positions)


class BartAttention(nn.Module):
    """Multi-head attention (self or cross) matching HF BartAttention."""

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, x: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, tgt_len, _ = hidden_states.size()
        is_cross = key_value_states is not None
        src = key_value_states if is_cross else hidden_states

        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(src)
        v = self.v_proj(src)

        q = self._shape(q, tgt_len, bsz)
        k = self._shape(k, k.size(1), bsz)
        v = self._shape(v, v.size(1), bsz)

        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # [bsz, heads, tgt, src]
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [bsz, heads, tgt, head_dim]

        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return self.out_proj(attn_output)


class BartEncoderLayer(nn.Module):
    def __init__(self, cfg: BartMnliConfig):
        super().__init__()
        self.self_attn = BartAttention(cfg.d_model, cfg.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, attention_mask=attention_mask)
        x = self.self_attn_layer_norm(residual + x)
        residual = x
        x = self.fc2(F.gelu(self.fc1(x)))
        x = self.final_layer_norm(residual + x)
        return x


class BartDecoderLayer(nn.Module):
    def __init__(self, cfg: BartMnliConfig):
        super().__init__()
        self.self_attn = BartAttention(cfg.d_model, cfg.decoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.encoder_attn = BartAttention(cfg.d_model, cfg.decoder_attention_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.decoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.decoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x, causal_mask, encoder_hidden, encoder_mask) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, attention_mask=causal_mask)
        x = self.self_attn_layer_norm(residual + x)

        residual = x
        x = self.encoder_attn(x, key_value_states=encoder_hidden, attention_mask=encoder_mask)
        x = self.encoder_attn_layer_norm(residual + x)

        residual = x
        x = self.fc2(F.gelu(self.fc1(x)))
        x = self.final_layer_norm(residual + x)
        return x


class BartEncoder(nn.Module):
    def __init__(self, cfg: BartMnliConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_scale = (cfg.d_model ** 0.5) if cfg.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.embed_positions = BartLearnedPositionalEmbedding(cfg.max_position_embeddings, cfg.d_model)
        self.layernorm_embedding = nn.LayerNorm(cfg.d_model)
        self.layers = nn.ModuleList([BartEncoderLayer(cfg) for _ in range(cfg.encoder_layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        seq_len = input_ids.size(1)
        x = self.embed_tokens(input_ids) * self.embed_scale
        x = x + self.embed_positions(seq_len, input_ids.device)
        x = self.layernorm_embedding(x)

        add_mask = _expand_mask(attention_mask, x.dtype, seq_len) if attention_mask is not None else None
        for layer in self.layers:
            x = layer(x, add_mask)
        return x


class BartDecoder(nn.Module):
    def __init__(self, cfg: BartMnliConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_scale = (cfg.d_model ** 0.5) if cfg.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.embed_positions = BartLearnedPositionalEmbedding(cfg.max_position_embeddings, cfg.d_model)
        self.layernorm_embedding = nn.LayerNorm(cfg.d_model)
        self.layers = nn.ModuleList([BartDecoderLayer(cfg) for _ in range(cfg.decoder_layers)])

    def forward(self, input_ids, encoder_hidden, encoder_attention_mask) -> torch.Tensor:
        seq_len = input_ids.size(1)
        x = self.embed_tokens(input_ids) * self.embed_scale
        x = x + self.embed_positions(seq_len, input_ids.device)
        x = self.layernorm_embedding(x)

        causal = _causal_mask(seq_len, x.dtype, x.device)
        enc_mask = (
            _expand_mask(encoder_attention_mask, x.dtype, seq_len)
            if encoder_attention_mask is not None else None
        )
        for layer in self.layers:
            x = layer(x, causal, encoder_hidden, enc_mask)
        return x


class BartModel(nn.Module):
    def __init__(self, cfg: BartMnliConfig):
        super().__init__()
        self.shared = nn.Embedding(cfg.vocab_size, cfg.d_model, cfg.pad_token_id)
        # Encoder/decoder token embeddings are tied to `shared`.
        self.encoder = BartEncoder(cfg, self.shared)
        self.decoder = BartDecoder(cfg, self.shared)

    def forward(self, input_ids, attention_mask, decoder_input_ids) -> torch.Tensor:
        encoder_hidden = self.encoder(input_ids, attention_mask)
        decoder_hidden = self.decoder(decoder_input_ids, encoder_hidden, attention_mask)
        return decoder_hidden


class BartClassificationHead(nn.Module):
    def __init__(self, cfg: BartMnliConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.classifier_dropout)
        self.out_proj = nn.Linear(cfg.d_model, cfg.num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class BartForSequenceClassification(nn.Module):
    """Reconstructed bart-large-mnli. ``forward`` returns 3-way NLI logits."""

    def __init__(self, cfg: BartMnliConfig = None):
        super().__init__()
        self.config = cfg or BartMnliConfig()
        self.model = BartModel(self.config)
        self.classification_head = BartClassificationHead(self.config)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        cfg = self.config
        if attention_mask is None:
            attention_mask = (input_ids != cfg.pad_token_id).long()

        decoder_input_ids = shift_tokens_right(input_ids, cfg.pad_token_id, cfg.decoder_start_token_id)
        hidden = self.model(input_ids, attention_mask, decoder_input_ids)

        # Pool the representation of the LAST </s> (EOS) token of each sequence.
        eos_mask = input_ids.eq(cfg.eos_token_id)
        if (eos_mask.sum(dim=1) == 0).any():
            raise ValueError("Every input must contain an EOS token for classification.")
        bsz = input_ids.size(0)
        sentence_rep = hidden[eos_mask, :].view(bsz, -1, hidden.size(-1))[:, -1, :]
        return self.classification_head(sentence_rep)

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask=None) -> torch.Tensor:
        return F.softmax(self.forward(input_ids, attention_mask), dim=-1)
