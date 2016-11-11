import glob
import time
from gensim import corpora, utils, models
import os

import logging

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("NgramGenerator")
logger.setLevel(logging.DEBUG)

class CorpusNTA(corpora.TextCorpus):

    stoplist = stopwords.words("english")
    lmtzr = WordNetLemmatizer()

    def file_contents(self, documents):
        for fn in documents:
            yield open(fn, 'r').read()

    def process_documents_to_bow(self):
        start_time = time.time()
        documents = [
            list(utils.tokenize(text, deacc=True, lower=True))
            for text in self.file_contents(self.input)
            ]
        elapsed_time = time.time() - start_time
        print('documents were loaded in {}ms'.format(int(elapsed_time * 1000)))

        return documents

    def prepare_bigrams(self):
        return models.phrases.Phrases(self.process_documents_to_bow())

    def get_texts(self):
        i = 0
        for fn in self.input:
            if i > 100:
                break
            i += 1
            text = open(fn, 'r').read()
            #yield [CorpusNTA.lmtzr.lemmatize(word) for word in list(utils.tokenize(text, deacc=True, lower=True)) if word not in CorpusNTA.stoplist]
            yield [word for word in list(utils.tokenize(text, deacc=True, lower=True)) if word not in CorpusNTA.stoplist]
            if i % 100 == 0:
                print("%d documents processed" % i)

    def __len__(self):
        """Define this so we can use `len(corpus)`"""
        if 'length' not in self.__dict__:
            print("caching corpus size (calculating number of documents)")
            self.length = sum(1 for doc in self.get_texts())
        return self.length

if __name__ == "__main__":
    #corpus_nta = CorpusNTA(glob.glob("documents_10/*"))
    # corpus_nta.save("corpus/documents_10_100.corpus")
    # corpus_nta = CorpusNTA.load("corpus/documents_10_100.corpus")
    # corpus_nta.dictionary.filter_extremes(no_below=5, no_above=0.1)

    # name = 'uci_corpus_100_10themes'
    # corpora.ucicorpus.UciCorpus.save_corpus("corpus/"+name, corpus_nta, corpus_nta.dictionary)
    # os.rename("corpus/"+name, "corpus/docword."+name+".txt")
    # os.rename("corpus/"+name+".vocab", "corpus/vocab."+name+".txt")


    # name = 'uci_corpus_100_10themes_bigram'
    #bigram = models.phrases.Phrases(corpus_nta)
    #bigram= corpus_nta.prepare_bigrams()
    #bigram.save("corpus/documents_10_100.bigram")
    bigram = models.phrases.Phrases.load("corpus/documents_10_100.bigram")
    # corpora.ucicorpus.UciCorpus.save_corpus("corpus/"+name, corpus_nta, corpus_nta.dictionary)
    # os.rename("corpus/"+name, "corpus/docword."+name+".txt")
    # os.rename("corpus/"+name+".vocab", "corpus/vocab."+name+".txt")
    print(bigram)

