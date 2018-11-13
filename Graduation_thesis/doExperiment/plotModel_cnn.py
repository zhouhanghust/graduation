from keras.utils import plot_model
import tensorflow as tf
from keras.layers import Dense, Conv1D, Activation, Dropout,\
                         GlobalMaxPooling1D, Concatenate,Embedding,Input

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

model = Model(input=wordsindex,outputs=predict)
plot_model(model, to_file='model.png',show_shapes=True)