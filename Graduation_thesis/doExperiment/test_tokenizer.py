# -*- coding: utf-8 -*-

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

text1 = 'some thing to eat'
text2 = 'some thing to drink'
text3 = 'some thing not happy'
text4 = 'happy birhtday some thing'
texts = [text1,text2,text3,text4]


tokenizer = Tokenizer(num_words=4) #num_words:None或整数,处理的单词数量。按频率来，排前num_words个词被保留。
tokenizer.fit_on_texts(texts)
print( tokenizer.word_counts) #[('some', 2), ('thing', 2), ('to', 2), ('eat', 1), ('drink', 1)]
print( tokenizer.word_index) #{'some': 1, 'thing': 2,'to': 3 ','eat': 4, drink': 5}
print( tokenizer.word_docs) #{'some': 2, 'thing': 2, 'to': 2, 'drink': 1,  'eat': 1}
print( tokenizer.index_docs) #{1: 2, 2: 2, 3: 2, 4: 1, 5: 1}

print("-----------------------------------------------")

# num_words=多少会影响下面的结果，列数=num_words
print( tokenizer.texts_to_sequences(texts)) #得到词索引[[1, 2, 3, 4], [1, 2, 3, 5]]
print( tokenizer.texts_to_matrix(texts))  # 矩阵化=one_hot

print("-----------------------------------------------")

MAX_SEQUENCE_LENGTH = 10
all_labels = [0,0,1]


tokenizer = Tokenizer(num_words=4) # 分词MAX_NB_WORDS
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) #受num_words影响
word_index = tokenizer.word_index # 词_索引
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  #将长度不足 10 的新闻用 0 填充（在前端填充）
labels = to_categorical(np.asarray(all_labels)) #最后将标签处理成 one-hot 向量，比如 6 变成了 [0,0,0,0,0,0,1,0,0,0,0,0,0]，
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
# Shape of data tensor: (81, 1000)  -- 81条数据
# Shape of label tensor: (81, 14)
print(data)
print(labels)









