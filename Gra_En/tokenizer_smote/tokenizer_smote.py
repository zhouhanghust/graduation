# from imblearn.combine import SMOTEENN
# from imblearn.over_sampling import SMOTE
import keras
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import numpy as np
import pickle
import random
# from imblearn.over_sampling import RandomOverSampler


with open("../cuted_unbalanced_data/texts.pkl", 'rb') as f:
    texts = pickle.load(f)

with open("../cuted_unbalanced_data/label.pkl", 'rb') as f:
    label = pickle.load(f)

#----------------------------


# 随机采样一部分数据
neg = np.asarray(label,dtype=np.int32) == 0
negInd = (np.where(neg))[0]
posInd = (np.where(np.asarray(label,dtype=np.int32)))[0]
choInd_pos = list(range(len(posInd)))
choInd_neg = list(range(len(negInd)))
random.shuffle(choInd_pos)
random.shuffle(choInd_neg)
posInd = posInd[choInd_pos][:30000]
negInd = negInd[choInd_neg][:30000]


inds = np.concatenate([posInd, negInd], axis=0)


newT = []
newL = []
for each in inds:
    newT.append(texts[each])
    newL.append(label[each])
print(len(newL))
print(sum(newL)/len(newL))

tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(newT)
sequences = tokenizer.texts_to_sequences(newT)

word_index = tokenizer.word_index

with open("./tokenizer_en.pkl", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''

with open("./tokenizer_en.pkl", 'rb') as f:
    tokenizer = pickle.load(f)
'''

data = pad_sequences(sequences, maxlen=64)
print(type(data))
# with open("./data_pad.pkl","wb") as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
with open("./data_pad.pkl","rb") as f:
    data = pickle.load(f)
'''

# ros = RandomOverSampler(random_state=0)
# X, y = ros.fit_sample(data, label)


with open("../en_ultimately/X.pkl","wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("../en_ultimately/y.pkl","wb") as handle:
    pickle.dump(newL, handle, protocol=pickle.HIGHEST_PROTOCOL)





