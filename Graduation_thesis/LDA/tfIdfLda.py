# -*- coding: utf-8 -*-
import pickle
import numpy as np
import gensim
from gensim import corpora, models
import pickle
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import math
import os

with open('../cutted_umbalanced_data/cutted_umbalanced_data_POS.pkl','rb') as f:
    data = pickle.load(f)

data = data.content_pos
result = []
for each in data:
    result.append(each.split())
data = result
num_docs = len(data)
dictionary = corpora.Dictionary(data)
#save dictionary
dictionary.save("./ldamodel" + os.sep + 'dictionary.dictionary')
#save corpus
corpus = [dictionary.doc2bow(doc) for doc in data]
corpora.MmCorpus.serialize("./ldamodel" + os.sep + 'corpus.mm', corpus)


def tfIdfLda(data,num_topics,iterations,workers,middatafolder):
    # middatafolder:保存模型的文件夹
    # num_topics:又用于区分模型的编号

    #load dictionary
    dictionary = corpora.Dictionary.load(middatafolder+'dictionary.dictionary')
    #load corpus
    corpus = corpora.MmCorpus(middatafolder + 'corpus.mm')

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lda_multi = models.ldamulticore.LdaMulticore(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_topics, iterations=iterations, workers=workers)
    #save lda_model
    lda_multi.save(middatafolder + 'lda_tfidf_%s.model'%num_topics)


if __name__ == "__main__":
    middatafolder = "./ldamodel"+os.sep
    for i in range(20):
        tfIdfLda(data,i+1,6000,1,middatafolder)
        print("the %sth model has been finished!"%(i+1))

