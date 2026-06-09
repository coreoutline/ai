import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Layer
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import warnings
import os
import pickle
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from itertools import chain
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, auc, confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

import nltk
from nltk.lm.preprocessing import pad_sequence
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
class Hestia():
    def __init__(self,vocab_size, maxlen):
        self.vocab_size = vocab_size  # Only consider the top 20k words
        self.maxlen = maxlen 

        self.embed_dim = 32  # Embedding size for each token
        self.num_heads = 2  # Number of attention heads
        self.ff_dim = 32  # Hidden layer size in feed forward network inside transformer

        inputs = Input(shape=(maxlen,))
        embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size, self.embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(20, activation="relu")(x)
        x = Dropout(0.1)(x)
        outputs = Dense(6, activation="softmax")(x)

        self.model = Model(inputs=inputs, outputs=outputs)
    
    def word_tokenize_text(self,x):
        return word_tokenize(x)
    
    def padding(self, x, max_length):
        max_length += 1
        return list(pad_sequence(x, n=(max_length-len(x)), pad_left=True, left_pad_symbol=0))
    
    def remove_punctuation(self,x):
        return "".join([i for i in x if i not in string.punctuation])
    
    def remove_stopwords(self,tokens):
        stopwords = nltk.corpus.stopwords.words('english')
        return [token for token in tokens if token not in stopwords]
    
    def labelencode(self,column):
        le = LabelEncoder()
        le.fit(column)
        self.labelencoder = le
    
    def word_map_tokenizer(self, text_arr):
        tokenizer = Tokenizer(num_words = 512, oov_token = '<00V>')
        tokenizer.fit_on_texts(text_arr)
        self.tokenizer = tokenizer

    def load_tokenizer(self, path):
        self.tokenizer = pickle.load(open(path, "rb"))

    def set_callbacks(self):
        callbacks = []
        earlystopping = EarlyStopping(monitor="loss",
                             patience=10,
                             min_delta=0,
                             mode='min',
                             restore_best_weights=False,
                             baseline=None,
                             verbose=0)
        callbacks.append(earlystopping)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                 factor=0.5,
                                                 patience=10,
                                                 min_lr=0.000001,
                                                 cooldown=5)

        callbacks.append(reduce_lr)
        checkpoint_path = "training_2/cp-{epoch:04d}.keras"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=False
            )

        callbacks.append(cp_callback)

        self.callbacks = callbacks

    def train(self, X_train, y_train, X_val, y_val):
        self.model.compile(optimizer=Adam(learning_rate=0.1), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        history = self.model.fit(X_train, y_train, 
                            batch_size=64, epochs=200, 
                            validation_data=(X_val, y_val),
                            callbacks=self.callbacks
                        )
        return self.model, history
    
    def load_model_weights(self, path):
        self.model.load_weights(path)
        print(self.model)
        return self.model

    def pop_layer(self):
        new_model = tf.keras.models.Model(inputs=self.model.input, 
                                  outputs=self.model.layers[-2].output)
        self.model = new_model

    def freeze_layers(self):
        for layer in self.model.layers:
            layer.trainable = False

    def append_layer(self, n_classes):
        x = self.model.output
        x = Dropout(0.1)(x)
        output = Dense(n_classes+1, activation='softmax')(x) 

        self.model = tf.keras.models.Model(inputs=self.model.input, outputs=output)