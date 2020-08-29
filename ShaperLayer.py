import tensorflow as tf


class ShaperLayer(tf.keras.layers.Layer):

    def __init__(self, input_dim):
        super(ShaperLayer, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        #self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        #self.total.assign_add()
        vector = tf.unstack(inputs)
        print(vector[0].shape)
        return tf.stack(vector)
