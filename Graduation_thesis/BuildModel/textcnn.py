# coding: utf-8
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv1D, Activation, Dropout,\
                         GlobalMaxPooling1D, Concatenate,Embedding
from keras import backend as K
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
K.set_learning_phase(1)

def load_data():

    x_data = np.load('../Chinese_ultimately/X.npy')
    y_data = np.load('../Chinese_ultimately/y.npy')

    X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_data, random_state=42, test_size=0.25)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()
ohe = OneHotEncoder(sparse=False, n_values=2)
ohe.fit(np.array([1, 0]).reshape(-1, 1))
y_train_ohe = ohe.transform(np.reshape(y_train,(-1,1)))
y_test_ohe = ohe.transform(np.reshape(y_test,(-1,1)))


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


with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x,labels=labels_oh))

tf.summary.scalar('loss', loss)


with tf.name_scope('accuracy'):
    match = tf.equal(tf.argmax(predict, axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))

tf.summary.scalar('accuracy', accuracy)


with tf.name_scope("train"):
    # train_op = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


init_op = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    merge_summary_loss = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'loss')])
    merge_summary_accuracy = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'accuracy')])
    summary_writer = tf.summary.FileWriter('./tb/', graph=sess.graph)
    K.set_session(sess)
    sess.run(init_op)
    epoch = 10
    batch_size = 256
    batches = len(y_train) // batch_size
    print(batches)
    loss_lst = [[],[]]
    acc_lst = [[],[]]
    for i in range(epoch):
        for j in range(batches):
            sess.run(train_op,
                feed_dict={wordsindex:X_train[j*batch_size:(j+1)*batch_size,:],labels_oh:y_train_ohe[j*batch_size:(j+1)*batch_size,:]})
            summary_loss = sess.run(merge_summary_loss,feed_dict={wordsindex:X_train,labels_oh:y_train_ohe})
            summary_accuracy = sess.run(merge_summary_accuracy,feed_dict={wordsindex:X_train,labels:y_train})
            summary_writer.add_summary(summary_loss, i*batches+j)
            summary_writer.add_summary(summary_accuracy, i * batches + j)
            loss_lst[0].append(sess.run(loss,feed_dict={wordsindex:X_train,labels_oh:y_train_ohe}))
            acc_lst[0].append(sess.run(accuracy, feed_dict={wordsindex:X_train,labels:y_train}))
            loss_lst[1].append(sess.run(loss,feed_dict={wordsindex:X_test,labels_oh:y_test_ohe}))
            acc_lst[1].append(sess.run(accuracy, feed_dict={wordsindex:X_test,labels:y_test}))

        print("the %sth epoch has been done!" % i)
    saver.save(sess,"model_save/model.ckpt",global_step=1024)
    summary_writer.close()
    with open("./LossAcc/loss.pkl","wb") as handle:
        pickle.dump(loss_lst,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open("./LossAcc/acc.pkl","wb") as handle:
        pickle.dump(acc_lst,handle,protocol=pickle.HIGHEST_PROTOCOL)

# with tf.Session() as sess:
#     K.set_session(sess)
#     saver.restore(sess, "./model_save/model.ckpt")
#
#     loss_ = sess.run(loss,feed_dict={wordsindex: X_test[:5000],labels:y_test[:5000]})
#     print(loss_)





