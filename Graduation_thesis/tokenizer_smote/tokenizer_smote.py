# -*- coding: utf-8 -*-
import pickle
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

with open('../cutted_umbalanced_data/cutted_umbalanced_data.pkl', 'rb') as f:
    df = pickle.load(f)

data = df['content'].tolist()
label = df['score'].tolist()

'''
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

word_index = tokenizer.word_index

with open("./tokenizer.pkl", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
with open("./tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

sequences = tokenizer.texts_to_sequences(data)

word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=64)


smote_k_neighbors = 5
smote = SMOTE(k_neighbors=smote_k_neighbors)
smote_enn = SMOTEENN(random_state=0, smote=smote)


X, y = smote_enn.fit_sample(data, label)

# 将X浮点型变整型
X = np.asarray(X, np.int32)
X = X.tolist()

print("-------------------------------------")
print(len(y))  # 11811
print(y.sum())  # 5438
print("-------------------------------------")

X = np.array(X)
y = np.array(y)

np.save('../Chinese_ultimately/X.npy',X)
np.save('../Chinese_ultimately/y.npy',y)