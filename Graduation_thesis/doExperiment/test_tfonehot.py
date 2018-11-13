import numpy as np
import tensorflow as tf

SIZE = 6
CLASS = 8
label1 = tf.constant([8, 9, 2, 3, 4, 0, 6, 7])
sess1 = tf.Session()
print('label1:', sess1.run(label1))
b = tf.one_hot(label1, CLASS, 1, 0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(b)
    print('after one_hot')
    print(sess.run(b))
