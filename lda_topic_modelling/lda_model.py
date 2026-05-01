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
from lda_topic_modelling.preprocess import preprocessing
warnings.filterwarnings("ignore")


def train_lda(num_topics):
    df = pd.read_csv("../Data/metrics.csv")
    data = preprocessing(df)
    id2word = corpora.Dictionary(data)
    pickle.dump(id2word, open('lda_topic_modelling/bagofwords.bow', 'wb'))
    corpus = [id2word.doc2bow(text) for text in data]
    pickle.dump(corpus, open('lda_topic_modelling/corpus.corpora', 'wb'))
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, update_every=1, chunksize=10,
                         passes=5, alpha='symmetric', iterations=100,
                         per_word_topics=True)
    pickle.dump(lda_model, open('lda_topic_modelling/lda.sav', 'wb'))
    df_topics_sents_keywords = format_topic_sentences(
        ldamodel=lda_model, corpus=corpus, texts=data)
    df_topics_sents_keywords['Review'] = df['Review'].values
    df_dominant_topic = df_topics_sents_keywords.reset_index()
    df_dominant_topic.to_csv('../Data/dominant_topic.csv')


def format_topic_sentences(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                cls_df = pd.DataFrame({
                    'Dominant_Topic': [int(topic_num)],
                    'Perc_Contribution': [round(prop_topic, 4)],
                    'Topic_Keywords': [topic_keywords]
                })
                sent_topics_df = pd.concat([sent_topics_df, cls_df], axis=0)
            else:
                break

    return sent_topics_df


def get_topics(sentence):
    df = pd.DataFrame()
    df['Review'] = [sentence]
    data = preprocessing(df)
    print(data)
    id2word = pickle.load(open('lda_topic_modelling/bagofwords.bow', 'rb'))
    new_doc_bow = [id2word.doc2bow(text) for text in data]
    lda_model = pickle.load(open('lda_topic_modelling/lda.sav', 'rb'))
    topic_distribution = lda_model.get_document_topics(new_doc_bow)
    print([x for x in topic_distribution])
    most_probable_topic = max([x[0] for x in topic_distribution])
    topic_id, topic_prob = most_probable_topic
    print(topic_id)
    print(topic_prob)
    df = pd.read_csv("../Data/dominant_topic.csv")
    contexts = df.loc[df['Dominant_Topic'] == topic_id]['Review'].values
    print(contexts)
    return contexts
