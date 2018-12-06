from keras.utils import plot_model
import tensorflow as tf
from keras.layers import Dense, Conv1D, Activation, Dropout, \
    GlobalMaxPooling1D, Concatenate, Embedding, Input, TimeDistributed, Bidirectional, GRU
from Graduation_thesis.BuildModel_CRNN_Attention.AttentionL import AttentionLayer
from keras.models import Model

with tf.name_scope('input'):
    wordsindex = Input(shape=(64,), dtype="float64")

with tf.name_scope('embedding'):
    wordsvector = Embedding(input_dim=9651,\
                            output_dim=256,\
                            input_length=64)(wordsindex)

x = Conv1D(256,3,strides=1,padding='valid',activation='relu')(wordsvector)
l_lstm = Bidirectional(GRU(256, return_sequences=True))(x)
l_dense = TimeDistributed(Dense(128))(l_lstm)  # 对句子中的每个词
l_att = AttentionLayer()(l_dense)

x = Dense(64)(l_att)
x = Dropout(0.5)(x)
x = Activation('relu')(x)
x = Dense(2)(x)
predict = Activation('softmax')(x)


model = Model(wordsindex, predict)


plot_model(model, to_file='model_crnn_attention.png',show_shapes=True)
