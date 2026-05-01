import tensorflow as tf
from tf.keras import Input, Model
from tf.keras.layers import Embedding, Dropout

from .pos_encoding import PositionalEncoding
from .encoding_layer import encoder_layer
def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
    """
    Comprises:
    Input embedding
    Positional encoding
    N of encoder layers
    """
    inputs = Input(shape=(None,), name="inputs")
    padding_mask = Input(shape=(1, 1, None), name="padding_mask")
    embeddings = Embedding(vocab_size, d_model)(embeddings)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    
    outputs = Dropout(rate=dropout)(embeddings)
    
    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name=f"encoder_layer_{i}"
        )([outputs, padding_mask])

    return Model([inputs, padding_mask], outputs=outputs, name=name)
