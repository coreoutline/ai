import tensorflow as tf
from tf.keras import Input, Model
from tf.keras.layers import Dropout, LayerNormalization, Dense
from .attention_transformer import MultiHeadAttention

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = Input(shape=(None, d_model), name="inputs")
    padding_mask = Input(shape=(1,1,None), name="padding_mask")

    attention  = MultiHeadAttention(
        d_model = d_model,
        num_heads = num_heads,
        name='attention'
    )
    ({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })

    attention = Dropout(rate=dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = Dense(units = units, activation='relu')(attention)
    outputs = Dense(units=d_model)(outputs)
    outputs = Dropout(rate=dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)

    return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
