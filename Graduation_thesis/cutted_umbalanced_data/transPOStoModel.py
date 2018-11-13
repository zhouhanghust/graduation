import pickle
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


with open('./cutted_umbalanced_data_POS.pkl','rb') as f:
    data = pickle.load(f)

data = data.content_pos.values


with open("./topicLst.pkl","rb") as f:
    topicLst = pickle.load(f)


with open("../tokenizer_smote/tokenizer.pkl", 'rb') as f:
    tokenizer = pickle.load(f)


def takeTopicDoc(docs, topicLst, topicInd):
    indLst = [i for i, x in enumerate(topicLst) if x == topicInd]
    return docs[indLst].tolist()


def makeTopicDocToModel(docs,topicLst,topicInd):
    mdata = takeTopicDoc(docs,topicLst,topicInd)
    sequences = tokenizer.texts_to_sequences(mdata)
    mdata = pad_sequences(sequences, maxlen=64)
    return mdata


if __name__ == "__main__":
    import numpy as np
    print(np.unique(topicLst))
    print(makeTopicDocToModel(data,topicLst,2))

