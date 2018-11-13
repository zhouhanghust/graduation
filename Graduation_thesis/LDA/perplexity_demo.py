from gensim.models import LdaModel
from gensim.corpora import Dictionary
import math

docs = [["a", "a", "b"],
        ["a", "c", "g"],
        ["c"],
        ["a", "c", "g"],
        ["e","a"],
        ["g","h"],
        ["a","e","e"],
        ["a", "a", "b"],
        ["a", "c", "g"],
        ["c"],
        ["a", "c", "g"],
        ["e", "a"],
        ["g", "h"],
        ["a", "e", "e"]]


dct = Dictionary(docs)
corpus = [dct.doc2bow(_) for _ in docs]
c_train, c_test = corpus[:2], corpus[2:]


ldamodel = LdaModel(corpus=c_train, num_topics=7, id2word=dct)
Perplexity_log = ldamodel.log_perplexity(c_test)
Perplexity = math.exp(-Perplexity_log)
print(Perplexity)
