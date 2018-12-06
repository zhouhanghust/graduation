# coding: utf-8
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv1D, Activation, Dropout, \
    GlobalMaxPooling1D, Concatenate, Embedding, Bidirectional, TimeDistributed, GRU
from keras import backend as K
from AttentionL import AttentionLayer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
K.set_learning_phase(1)


def load_data():
    with open("../en_ultimately/X.pkl","rb") as f:
        x_data = pickle.load(f)
    with open("../en_ultimately/y.pkl","rb") as f:
        y_data = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_data, random_state=42, test_size=0.25)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()
ohe = OneHotEncoder(sparse=False, n_values=2)
ohe.fit(np.array([1, 0]).reshape(-1, 1))
y_train_ohe = ohe.transform(np.reshape(y_train,(-1,1)))
y_test_ohe = ohe.transform(np.reshape(y_test,(-1,1)))


with open('../tokenizer_smote/tokenizer_en.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

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
    x = Conv1D(256, 3, strides=1, padding='valid', activation='relu')(wordsvector)
    l_lstm = Bidirectional(GRU(256, return_sequences=True))(x)
    l_dense = TimeDistributed(Dense(128))(l_lstm)  # 对句子中的每个词
    l_att = AttentionLayer()(l_dense)
    x = Dense(64)(l_att)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    predict = Activation('softmax')(x)


with tf.name_scope('predict'):
    pred = tf.argmax(predict, axis=1)
    match = tf.equal(tf.argmax(predict, axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))


with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x,labels=labels_oh))


saver = tf.train.Saver()


with tf.Session() as sess:
    K.set_session(sess)
    saver.restore(sess, "./model_save/model.ckpt-150")

    inbatch_size = 2048
    inbatches_test = len(y_test) // inbatch_size

    temp_pred1 = []
    predict_prob = []

    for jj in range(inbatches_test):
        temp_pred1.extend(sess.run(pred, feed_dict={
            wordsindex: X_test[jj * inbatch_size:(jj + 1) * inbatch_size, :]}))
        pred_prob = sess.run(predict, feed_dict={wordsindex: X_test[jj * inbatch_size:(jj + 1) * inbatch_size, :]})
        predict_prob.extend([each[1] for each in pred_prob])
        print("the %sth inbatches has been done! total: %s"%(jj+1,inbatches_test))

    if (jj+1)*inbatch_size < len(X_test):
        diff = len(X_test) - (jj+1)*inbatch_size
        temp_pred1.extend(sess.run(pred, feed_dict={
            wordsindex: X_test[(jj + 1) * inbatch_size:, :]}))
        pred_prob = sess.run(predict, feed_dict={wordsindex: X_test[(jj + 1) * inbatch_size:, :]})
        predict_prob.extend([each[1] for each in pred_prob])
        print("all of the inbatches has been done! total")

    print(sum(temp_pred1)/len(temp_pred1))
    tofile = []
    tofile.append(temp_pred1)
    tofile.append(y_test)
    print(len(tofile[0])/len(tofile[1]))

    with open("./predAndy_test.pkl","wb") as f:
        pickle.dump(tofile, f, pickle.HIGHEST_PROTOCOL)

    predict_prob_label = [predict_prob, y_test]
    with open("./predict_prob_label.pkl", "wb") as f:
        pickle.dump(predict_prob_label, f, pickle.HIGHEST_PROTOCOL)