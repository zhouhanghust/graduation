from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import keras
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import numpy as np
import pickle
import random

with open("../cuted_unbalanced_data/texts.pkl", 'rb') as f:
    texts = pickle.load(f)

with open("../cuted_unbalanced_data/label.pkl", 'rb') as f:
    label = pickle.load(f)


# tokenizer = Tokenizer(num_words=None)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
#
# word_index = tokenizer.word_index
#
# with open("./tokenizer_en.pkl", 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
#----------------------------

with open("./tokenizer_en.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

# 随机采样一部分数据来测试
index = list(range(len(label)))[:100]
texts = [texts[ind] for ind in index]
label = [label[ind] for ind in index]





sequences = tokenizer.texts_to_sequences(texts)

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

np.save('../en_ultimately/X.npy',X)
np.save('../en_ultimately/y.npy',y)
