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

with open('../cutted_umbalanced_data/cutted_umbalanced_data.pkl','rb') as f:
    data = pickle.load(f)

data = data.content
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

    '''
    thetas = [lda_multi[c] for c in corpus_tfidf]
    font = FontProperties(fname="simhei.ttf", size=18)
    
    
    # plot numOfDocAndTopic
    lst = []
    for each in thetas:
        temp = sorted(each, key=lambda x: x[1])
        lst.append(str(temp[-1][0]))
    se = pd.DataFrame({'data':lst})
    var = se.groupby('data').apply(len)

    var.plot(kind='bar')
    plt.ylabel('文档数',fontproperties=font)
    plt.xlabel('主题数',fontproperties=font)
    plt.show()
    # plt.tight_layout()
    # plt.savefig('./pic_idf.png',dpi=250)
    

    topic_detail = lda_multi.print_topics(num_topics=num_topics, num_words=15)
    doc_lda = lda_multi.get_document_topics(corpus_tfidf)


    def get_document_topic(doc_lda):
        index = -1
        temp = 0
        for item in doc_lda:
            if temp < item[1] :
                index = item[0]
                temp = item[1]
        return index


    topic_pre=[] # 得到corpus中每篇文档对应的主题编号
    for each in doc_lda:
        index = get_document_topic(each)
        topic_pre.append(index)

    '''


if __name__ == "__main__":
    middatafolder = "./ldamodel"+os.sep
    for i in range(20):
        tfIdfLda(data,i+1,6000,1,middatafolder)