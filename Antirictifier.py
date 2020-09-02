import tensorflow as tf
import numpy as np

class Antirectifier(tf.keras.layers.Layer):
    def __init__(self):
        super(Antirectifier, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((3,)), trainable=False)

    def call(self, inputs):
        #self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return inputs

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(Antirectifier, self).get_config()
        return dict(list(base_config.items()))