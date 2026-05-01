# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# from tensorflow.keras.layers import Embedding, Input, Dropout, Conv1D, GlobalMaxPooling1D, Dense
# from tensorflow.keras import Model, Input
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.utils import pad_sequence

# import re
# import string

# from importlib.metadata import version

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
# print("tiktoken version:", version("tiktoken"))


class CoreDataset(Dataset):
    def __init__(self, txt, tokenizer, max_len, step):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_len, step):
            input_chunk = token_ids[i:i+max_len]
            target_chunk = token_ids[i+1: i+max_len+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        self.input_ids = torch.stack(self.input_ids)
        self.target_ids = torch.stack(self.target_ids)
        self.input_ids.to("cpu")
        self.target_ids.to("cpu")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


# class Preparation:
#     """
#     Use byte pair encoding
#     """
#     def __init__(self, max_features, text, embedding_dim, max_length):
#         self.text = text
#         self.MAX_FEATURES = max_features
#         self.EMBEDDING_DIM = embedding_dim
#         self.MAX_LENGTH  = max_length
    
#     def remove_punctuation(self):
#         return "".join([i for i in self.text if i not in string.punctuation])
    
#     def lower_case(self):
#         return self.text.lower()
    
#     def word_tokenize_(self):
#         return nltk.word_tokenize(self.text)
    
#     def sent_tokenize_(self):
#         START_TOKEN = "+==="
#         END_TOKEN = "===+"
#         result = [(START_TOKEN + i + END_TOKEN) for i in nltk.sent_tokenize(self.text) if len(START_TOKEN + i + END_TOKEN) < self.MAX_LENGTH] 
#         return result
    
#     def remove_stopwords(self, tokens):
#         stopwords = nltk.corpus.stopwords.words('english')
#         return [token for token in tokens if token not in stopwords]
    
#     def stem_words(self, tokens):
#         porter_stemmer = PorterStemmer()
#         return [porter_stemmer.stem(token) for token in tokens]
    
#     def lemmatize_words(self, tokens):
#         wordnet_lemmatizer = WordNetLemmatizer()
#         return [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    
#     def remove_urls(self, text):
#         return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    
#     def text_vectorization(self, text_arr):
#         vectorize_layer = TextVectorization(
#             max_tokens = self.MAX_FEATURES,
#             output_mode = 'int',
#             output_sequence_length = 500
#         )
#         vectorize_layer.adapt(text_arr, batch_size=64)
        
#         X_train_padded =  vectorize_layer(text_arr)
#         X_train_padded = X_train_padded.np()
#         X_train_padded = np.reshape(X_train_padded, (np.shape(X_train_padded)[0], np.shape(X_train_padded)[1], 1 ))

#         return X_train_padded
    
#     def word_map_tokenizer(self, text_arr):
#         tokenizer = Tokenizer(num_words = 100, oov_token = '<00V>')
#         self.tokenizer = tokenizer.fit_on_texts(text_arr)

#     def get_id_seq(self, text):
#         return self.tokenizer.texts_to_sequences([text])
    
#     def get_text_seq(self, id_seq):
#         return self.tokenizer.sequences_to_text([id_seq])
    
#     def pad_seq_(self, )
    




    
    


# Here is the call for the data loader
def create_dataloader_v1(txt, batch_size=4, max_len=256,step=128, shuffle=True, drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = CoreDataset(txt, tokenizer, max_len, step)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader, tokenizer


# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
#     raw_text = raw_text.replace("<|endoftext|>", " <|endoftext|> ")
#     dataloader = create_dataloader_v1(
#     raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
#     data_iter = iter(dataloader)
#     first_batch = next(data_iter)
#     print(first_batch)



# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# print("Inputs:\n", inputs)
# print("\nTargets:\n", targets)
