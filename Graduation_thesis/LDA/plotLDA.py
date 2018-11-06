# -*- coding: utf-8 -*
from gensim import corpora, models
import pickle
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import os


middatafolder = "./ldamodel"+os.sep

# load dictionary
dictionary = corpora.Dictionary.load(middatafolder + 'dictionary.dictionary')
# load corpus
corpus = corpora.MmCorpus(middatafolder + 'corpus.mm')
# load model
lda_multi = models.ldamodel.LdaModel.load(middatafolder + 'lda_tfidf_13.model')

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


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


topic_detail = lda_multi.print_topics(num_topics=13, num_words=20)
print(topic_detail)
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

