import pickle
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def takeTopicDoc(docs, topicLst, topicInd):
    indLst = [i for i, x in enumerate(topicLst) if x == topicInd]
    return docs[indLst].tolist()


def makeTopicDocToModel(docs,topicLst,topicInds,tokenizer):
    mdata = []
    for topic in topicInds:
        mdata.extend(takeTopicDoc(docs,topicLst,topic))
    sequences = tokenizer.texts_to_sequences(mdata)
    mdata = pad_sequences(sequences, maxlen=64)
    return mdata


if __name__ == "__main__":
    import numpy as np

    with open('./balanced_data.pkl', 'rb') as f:
        data = pickle.load(f)


    with open("./topicLst.pkl", "rb") as f:
        topicLst = pickle.load(f)

    with open("../tokenizer_smote/tokenizer.pkl", 'rb') as f:
        tokenizer = pickle.load(f)
    print(len(topicLst))
    print(np.unique(topicLst))
    data = np.asarray(data)
    print(makeTopicDocToModel(data,topicLst,[0],tokenizer))

