import tensorflow as tf


class ComputeSum(tf.keras.layers.Layer):

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        #self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        #self.total.assign_add()
        return tf.constant([1.,0.])
