# coding: utf-8
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, \
    Embedding, Bidirectional, TimeDistributed, GRU
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
max_features = len(word_index)+1  #9651



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
    l_lstm = Bidirectional(GRU(256, return_sequences=True))(wordsvector)
    l_dense = TimeDistributed(Dense(256))(l_lstm)  # 对句子中的每个词
    l_att = AttentionLayer()(l_dense)
    x = Dense(64)(l_att)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    predict = Activation('softmax')(x)


with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x,labels=labels_oh))

#tf.summary.scalar('loss', loss)


with tf.name_scope('accuracy'):
    match = tf.equal(tf.argmax(predict, axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))

#tf.summary.scalar('accuracy', accuracy)


with tf.name_scope("train"):
    # train_op = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
#    merge_summary_loss = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'loss')])
#    merge_summary_accuracy = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'accuracy')])
#    summary_writer = tf.summary.FileWriter('./tb/', graph=sess.graph)
    K.set_session(sess)
    sess.run(init_op)
    epoch = 1
    batch_size = 4096
    batches = len(y_train) // batch_size
    print(batches)

    inbatch_size = 4096
    inbatches_test = len(y_test) // inbatch_size
    inbatches_train = len(y_train) // inbatch_size

    loss_lst = [[],[]]
    acc_lst = [[],[]]
    for i in range(epoch):
        for j in range(batches):
            sess.run(train_op,
                feed_dict={wordsindex:X_train[j*batch_size:(j+1)*batch_size,:],labels_oh:y_train_ohe[j*batch_size:(j+1)*batch_size,:]})
#            summary_loss = sess.run(merge_summary_loss,feed_dict={wordsindex:X_train,labels_oh:y_train_ohe})
#            summary_accuracy = sess.run(merge_summary_accuracy,feed_dict={wordsindex:X_train,labels:y_train})
#            summary_writer.add_summary(summary_loss, i*batches+j)
#            summary_writer.add_summary(summary_accuracy, i * batches + j)

            temp_loss0 = []
            temp_loss1 = []
            temp_acc0 = []
            temp_acc1 = []

            for jj in range(inbatches_train):
                temp_loss0.append(inbatch_size * sess.run(loss, feed_dict={
                    wordsindex: X_train[jj * inbatch_size:(jj + 1) * inbatch_size, :],
                    labels_oh: y_train_ohe[jj * inbatch_size:(jj + 1) * inbatch_size, :]}))
                temp_acc0.append(inbatch_size * sess.run(accuracy, feed_dict={
                    wordsindex: X_train[jj * inbatch_size:(jj + 1) * inbatch_size, :],
                    labels: y_train[jj * inbatch_size:(jj + 1) * inbatch_size]}))

            for jj in range(inbatches_test):
                temp_loss1.append(inbatch_size * sess.run(loss, feed_dict={
                    wordsindex: X_test[jj * inbatch_size:(jj + 1) * inbatch_size, :],
                    labels_oh: y_test_ohe[jj * inbatch_size:(jj + 1) * inbatch_size, :]}))
                temp_acc1.append(inbatch_size * sess.run(accuracy, feed_dict={
                    wordsindex: X_test[jj * inbatch_size:(jj + 1) * inbatch_size, :],
                    labels: y_test[jj * inbatch_size:(jj + 1) * inbatch_size]}))

            loss_lst[0].append(sum(temp_loss0) / (inbatches_train * inbatch_size))
            loss_lst[1].append(sum(temp_loss1) / (inbatches_test * inbatch_size))
            acc_lst[0].append(sum(temp_acc0) / (inbatches_train * inbatch_size))
            acc_lst[1].append(sum(temp_acc1) / (inbatches_test * inbatch_size))

            iterations = i*batches + j + 1
            remainder = (iterations) % 30
            if remainder == 0:
                saver.save(sess, "model_save/model.ckpt", global_step=iterations)
            print("the %sth batches has been done!" % iterations)
        print("the %sth epoch has been done!" % i)
#    summary_writer.close()
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
