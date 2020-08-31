import tensorflow as tf
import numpy as np

class ShaperLayer(tf.keras.layers.Layer):

    def concat_result(self, __array):

        convert_dict = {0: 1, 1: 0, 2: -1}

        vector_up = np.array(__array).astype(np.float32)[0:3]
        vector_none = np.array(__array).astype(np.float32)[3:6]
        vector_down = np.array(__array).astype(np.float32)[6:9]

        max_index_up = convert_dict[np.argmax(vector_up, axis=0)]
        max_index_none = convert_dict[np.argmax(vector_none, axis=0)]
        max_index_down = convert_dict[np.argmax(vector_down, axis=0)]

        calc_value = abs(max_index_none) * (max_index_up + max_index_down + max_index_none)

        print (calc_value)

        if (calc_value == 0):
            return np.array([0, 1, 0])
        elif (calc_value >= 1):
            return np.array([1, 0, 0])
        elif (calc_value <= -1):
            return np.array([0, 0, 1])

    def __init__(self, input_dim):
        super(ShaperLayer, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        return tf.convert_to_tensor(self.concat_result(inputs.numpy()))

