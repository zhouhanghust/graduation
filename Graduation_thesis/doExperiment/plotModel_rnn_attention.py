from keras.utils import plot_model
import tensorflow as tf
from keras.layers import Dense, Conv1D, Activation, Dropout, \
    GlobalMaxPooling1D, Concatenate, Embedding, Input, TimeDistributed, Bidirectional, GRU
from BuildModel_CRNN_Attention.AttentionL import AttentionLayer
from keras.models import Model

with tf.name_scope('input'):
    # wordsindex = tf.placeholder(tf.int64,shape=(None,64))
    wordsindex = Input(shape=(64,), dtype="float64")
    # labels = tf.placeholder(tf.int64,shape=(None,))
    # labels_oh = tf.one_hot(labels, 2, 1, 0)
    # labels_oh = tf.placeholder(tf.float64,shape=(None,2))
with tf.name_scope('embedding'):
    wordsvector = Embedding(input_dim=9651,\
                            output_dim=256,\
                            input_length=64)(wordsindex)


l_lstm = Bidirectional(GRU(256, return_sequences=True))(wordsvector)
l_dense = TimeDistributed(Dense(100))(l_lstm)  # 对句子中的每个词
l_att = AttentionLayer()(l_dense)
model = Model(wordsindex, l_att)


plot_model(model, to_file='model_rnn_attention.png',show_shapes=True)