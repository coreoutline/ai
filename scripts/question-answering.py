from transformers import pipeline
import pandas as pd
from lda_topic_modelling.lda_model import get_topics, train_lda


train_lda(50)
qa_model = pipeline("question-answering")


def answer(question):
    contexts = get_topics(question)
    possible_answers = []
    for context in contexts:
        possible_answers.append(qa_model(question=question, context=context))

    print(possible_answers)


answer("What is customer acquisition cost?")
