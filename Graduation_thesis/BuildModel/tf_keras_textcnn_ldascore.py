# coding: utf-8
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv1D, Activation, Dropout,\
                         GlobalMaxPooling1D, Concatenate,Embedding
from keras import backend as K
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Graduation_thesis.cutted_umbalanced_data.transPOStoModel import makeTopicDocToModel


with open("../cutted_umbalanced_data/topicLst.pkl", "rb") as f:
    topicLst = pickle.load(f)

with open('../tokenizer_smote/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('../cutted_umbalanced_data/balanced_data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('../cutted_umbalanced_data/balanced_data_label.pkl', 'rb') as f:
    score = pickle.load(f)


print(sum(score))
print(len(score))
print(sum(score)/len(score))

data = np.asarray(data)
feed_data = makeTopicDocToModel(data,topicLst,[3],tokenizer)


word_index = tokenizer.word_index
max_features = len(word_index)+1

with tf.name_scope('input'):
    wordsindex = tf.placeholder(tf.int64,shape=(None,64))
    labels = tf.placeholder(tf.int64,shape=(None,))
    # labels_oh = tf.one_hot(labels, 2, 1, 0)
    labels_oh = tf.placeholder(tf.float64,shape=(None,2))
with tf.name_scope('embedding'):
    wordsvector = Embedding(input_dim=max_features,\
                            output_dim=256,\
                            input_length=64)(wordsindex)


with tf.name_scope("NN"):
    regions = [3,4,5]
    inputs = []
    convs = []
    for region in regions:
        x = Conv1D(250,region,strides=1,padding='valid',activation='relu')(wordsvector)
        x = GlobalMaxPooling1D()(x)
        convs.append(x)

    x = Concatenate()(convs)
    x = Dense(250)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    predict = Activation('softmax')(x)


with tf.name_scope('predict'):
    pred = tf.argmax(predict, axis=1)
    match = tf.equal(tf.argmax(predict, axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))

saver = tf.train.Saver()


with tf.Session() as sess:
    K.set_session(sess)
    saver.restore(sess, "./model_save/model.ckpt-60")

    pred_ = sess.run(pred, feed_dict={wordsindex: feed_data})
    print(sum(pred_)/len(pred_))



