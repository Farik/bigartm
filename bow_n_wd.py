import glob
import time

import re

import artm
from gensim import utils

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk

def file_contents(documents):
    for fn in documents:
        yield open(fn, 'r').read()

def build_texts(input):
    start_time = time.time()
    documents = [
        #list(utils.tokenize(text, deacc=True, lower=True))
        text for text in file_contents(input)
        ]
    elapsed_time = time.time() - start_time
    print('documents were loaded in {}ms'.format(int(elapsed_time * 1000)))

    return documents


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

limit = 3
documents = build_texts(glob.glob("documents_10/*")[0:limit])

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_only, ngram_range=(1,3))


print("Extracting tf-idf features for %d documents..." % limit)
t0 = time.time()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print("done in %0.3fs." % (time.time() - t0))

print("Extracting tf features %d documents ..." % limit)
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=10,
                                stop_words='english')
t0 = time.time()
tf = tf_vectorizer.fit_transform(documents)
print("done in %0.3fs." % (time.time() - t0))


batch_vectorizer = artm.BatchVectorizer(data_format='bow_n_wd', n_wd=tfidf_matrix.vocabulary, vocabulary=tfidf_matrix.vocabulary)