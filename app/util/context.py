from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from itertools import chain
import string
import numpy as np

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    return tokens

def get_samples_bm25(df, cluster, num, bm25, test):
    tokenized_query = preprocess(test)

    doc_scores = bm25.get_scores(tokenized_query)
    x = np.argpartition(doc_scores, -num)[-num:]
    sampled_data = df.iloc[x]
    return list(chain.from_iterable(zip(sampled_data['query'], sampled_data['utterance'])))