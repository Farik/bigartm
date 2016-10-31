import gensim
import gensim
import glob
import time
from gensim import utils, corpora
from gensim.corpora import ucicorpus

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("NgramGenerator")
logger.setLevel(logging.DEBUG)

class CorpusNTA(corpora.TextCorpus):

    stoplist = set('for a of the and to in on'.split())

    def get_texts(self):
        i=0
        for fn in self.input:
            text = open(fn, 'r').read()
            yield [word for word in list(utils.tokenize(text, deacc=True, lower=True)) if word not in CorpusNTA.stoplist]

    def __len__(self):
        """Define this so we can use `len(corpus)`"""
        if 'length' not in self.__dict__:
            print("caching corpus size (calculating number of documents)")
            self.length = sum(1 for doc in self.get_texts())
        return self.length

if __name__ == "__main__":
    corpus_nta = CorpusNTA(glob.glob("documents/*"))
    ucicorpus.UciCorpus.save_corpus("corpus/uci_corpus", corpus_nta, corpus_nta.dictionary)

