import re
import nltk
import numpy as np
import pandas as pd
from pprint import pprint
import pickle
import gensim
import spacy
import logging
import warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

stopwords = nltk.corpus.stopwords.words('english')


def remove_emails_nl_quotes(sentence):
    if (sentence is np.nan):
        return
    sentence = re.sub('\S*@\S*\s?', '', sentence)
    sentence = re.sub('\s+', ' ', sentence)
    sentence = re.sub("\'", "", sentence)
    sentence = gensim.utils.simple_preprocess(str(sentence), deacc=True)
    return sentence


def word_tokenization(sentence):
    return [token for token in sentence if token not in stopwords]


def preprocessing(df):
    df.dropna(inplace=True)
    df['Preprocessed'] = df['Review'].apply(remove_emails_nl_quotes)
    df['No_Stopwords'] = df['Preprocessed'].apply(word_tokenization)
    data_words = df['No_Stopwords'].values
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)

    texts = data_words
    texts = [bigram_model[doc] for doc in texts]
    texts = [trigram_model[bigram_model[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load("en_core_web_sm")
    allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
        texts_out = [[word for word in simple_preprocess(
            str(doc)) if word not in stopwords] for doc in texts_out]

    data_ready = texts_out
    return data_ready
