import tensorflow as tf
import numpy as np


class ConcatLayer(tf.keras.layers.Layer):

    def __get_max_index(self, vector):

        winner = tf.where(vector == tf.math.reduce_max(vector))
        if (winner.shape[0] > 1):
            return 0
        else:
            return self.convert_dict[tf.math.argmax(vector).numpy()]

    def __find_best_data(self, up, none, down, idx):

        max_index_array = tf.math.argmax(tf.concat([tf.slice(up, [idx], [1]), tf.slice(none, [idx], [1]),
                                        tf.slice(down, [idx], [1])], 0)).numpy()
        if(max_index_array == 1):
            return none

        elif (max_index_array == 0):
            return up

        else:
            return down

    def __remove_ex_data(self, vector, max_idex, calc_value):

        if (calc_value > 0 and max_idex > 0):
            return vector

        elif (calc_value < 0 and max_idex < 0):
            return vector

        elif (calc_value == 0 and max_idex == 0):
            return vector

        return tf.constant([0, 0, 0], dtype=float)

    def __concat_result(self, vector):

        vector_up = tf.slice(vector, [0], [3])
        vector_none = tf.slice(vector, [3], [3])
        vector_down = tf.slice(vector, [6], [3])

        max_index_up = self.__get_max_index(vector_up)
        max_index_none = self.__get_max_index(vector_none)
        max_index_down = self.__get_max_index(vector_down)

        calc_value = abs(max_index_none) * (max_index_up + max_index_down + max_index_none)

        vector_up = self.__remove_ex_data(vector_up, max_index_up, calc_value)
        vector_none = self.__remove_ex_data(vector_none, max_index_none, calc_value)
        vector_down = self.__remove_ex_data(vector_down, max_index_down, calc_value)

        if (calc_value == 0):
            return self.__find_best_data(vector_up, vector_none, vector_down, 1)
        elif (calc_value >= 1):
            return self.__find_best_data(vector_up, vector_none, vector_down, 0)
        elif (calc_value <= -1):
            return self.__find_best_data(vector_up, vector_none, vector_down, 2)

    def __wrapper(self, inputs):

        for vector in tf.data.Dataset.from_tensor_slices(inputs):
            x = self.__concat_result(vector)
            self.total = tf.concat([self.total, tf.reshape(x, [1, 3])], 0)
        return self.total

    def call(self, inputs, **kwargs):
        return self.__wrapper(inputs)

    def __init__(self):
        super(ConcatLayer, self).__init__()
        self.convert_dict = {0: 1, 1: 0, 2: -1}


    def build(self, input_shape):
        self.total = tf.Variable(np.empty((0, 3), dtype=np.float32))
        return super(ConcatLayer, self).build(input_shape)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(ConcatLayer, self).get_config()
        return dict(list(base_config.items()))





