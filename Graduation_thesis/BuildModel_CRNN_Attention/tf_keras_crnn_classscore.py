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

    x_data = np.load('../Chinese_ultimately/X.npy')
    y_data = np.load('../Chinese_ultimately/y.npy')

    X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_data, random_state=42, test_size=0.25)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()
# ohe = OneHotEncoder(sparse=False, n_values=2)
# ohe.fit(np.array([1, 0]).reshape(-1, 1))
# y_train_ohe = ohe.transform(np.reshape(y_train,(-1,1)))
# y_test_ohe = ohe.transform(np.reshape(y_test,(-1,1)))


with open('../tokenizer_smote/tokenizer.pkl', 'rb') as handle:
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

saver = tf.train.Saver()


with tf.Session() as sess:
    K.set_session(sess)
    saver.restore(sess, "./model_save/model.ckpt-60")

    pred_ = sess.run(pred, feed_dict={wordsindex: X_test})
    accuracy_ = sess.run(accuracy, feed_dict={wordsindex: X_test,labels:y_test})
    pred_prob = sess.run(predict, feed_dict={wordsindex: X_test})
    print(sum(y_test) / len(y_test))
    print(sum(pred_)/len(pred_))
    print("-----acc-----")
    print(accuracy_)

predict_prob = [each[1] for each in pred_prob]
predict_prob_label = [predict_prob,y_test]
with open("./predict_prob_label.pkl","wb") as f:
    pickle.dump(predict_prob_label, f, pickle.HIGHEST_PROTOCOL)



from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def my_confusion_matrix(y_true, y_pred):
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)
    total = 0
    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]:
            total += 1
    print("Accuracy: %s"%(total/len(y_pred)))
    print("第一行是y_true为0，第二行是y_true为1，第一列是y_pred为0，第二列是y_pred为1")
    print(conf_mat)


my_confusion_matrix(y_test,pred_)


print("classification_report(left: labels):")
print(classification_report(y_test, pred_))
print("第二行 标签为1 的是需要的数据")
# presicion=TP/(TP+FP) recall = TP/(TP+FN)
'''
FN：False Negative,被判定为负样本，但事实上是正样本。
FP：False Positive,被判定为正样本，但事实上是负样本。
TN：True Negative,被判定为负样本，事实上也是负样本。
TP：True Positive,被判定为正样本，事实上也是证样本。
'''

