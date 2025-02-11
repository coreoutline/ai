# import tensorflow as tf
# from tf.nn import softmax
# from tf.keras.layers import Dense

import torch
from torch import nn

# class MultiHeadAttention:
#     """
#     Comprises four parts
#     1) Linear Layers into split heads
#     2) Scaled dot-product attention
#     3) Concatenation of heads
#     4) Final linear layer

#     Input of MultiHead attention block is a dictionary of query, key and value
#     Call using the call function
#     """
#     def __init__(self, d_model, num_heads,name="multi_head_attention"):
#         super(MultiHeadAttention, self).__init__(name=name)
#         self.num_heads = num_heads
#         self.d_model = d_model

#         assert d_model % self.num_heads == 0

#         self.depth = d_model // self.num_heads

#         self.query_dense = Dense(units=d_model)
#         self.key_dense = Dense(units=d_model)
#         self.value_dense = Dense(units=d_model)

#         self.dense = Dense(units=d_model)


#     def scaled_dot_product_attention(query, key, value, mask):
#         matmul_qk = tf.matmul(query, key, transpose_b=True)
#         depth = tf.cast(tf.shape(key)[-1], tf.float32)
#         logits = matmul_qk / tf.math.sqrt(depth)

#         if mask is not None:
#             logits += (mask * -1e9)

#         attention_weights = softmax(logits, axis=-1)

#         return tf.matmul(attention_weights, value)
    
#     def split_heads(self, inputs, batch_size):
#         inputs = tf.reshape(
#             inputs,
#             shape=(batch_size, -1, self.num_heads, self.depth)
#         )

#         return tf.transpose(inputs, perm=[0,2,1,3])
    
#     def call(self, inputs):
#         query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
#         batch_size = tf.shape(query)[0]

#         query = self.query_dense(query)
#         key = self.key_dense(key)
#         value = self.value_dense(value)

#         query = self.split_heads(query, batch_size = batch_size)
#         key = self.split_heads(query, batch_size=batch_size)
#         value = self.split_heads(value, batch_size=batch_size)

#         scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)
#         scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])
#         concat_attention = tf.reshape(scaled_attention,
#                                       (batch_size, -1, self.d_model))
#         return self.dense(concat_attention)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
        dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                        diagonal=1
            )
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_( self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(  attn_scores / keys.shape[-1]**0.5, dim=-1 )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), """d_out must be divisible by num_heads"""
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim )
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view( b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec