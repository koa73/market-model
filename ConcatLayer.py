import tensorflow as tf
import numpy as np

class ConcatLayer(tf.keras.layers.Layer):

    def __get_max_index(self, vector):

        winner = np.argwhere(vector == np.amax(vector))
        if (winner.size > 1):
            return 0
        else:
            return self.convert_dict[winner[0][0]]

    def __find_best_data(self, up, none, down, idx):

        tmp = np. array([up[self.convert_dict[idx]], none[self.convert_dict[idx]], down[self.convert_dict[idx]]], dtype='float32')
        max_index_array = np.argmax(tmp, axis=0)

        if(max_index_array == 1):
            return none
        elif (max_index_array == 0):
            return up
        else:
            return down



    def __concat_result(self, __array):

        vector_up = np.array(__array).astype(np.float32)[0:3]
        vector_none = np.array(__array).astype(np.float32)[3:6]
        vector_down = np.array(__array).astype(np.float32)[6:9]

        max_index_up = self.__get_max_index(vector_up)
        max_index_none = self.__get_max_index(vector_none)
        max_index_down = self.__get_max_index(vector_down)

        calc_value = abs(max_index_none) * (max_index_up + max_index_down )

        return self.__find_best_data(vector_up,vector_none,vector_down, calc_value)

    def __init__(self):
        super(ConcatLayer, self).__init__()
        self.convert_dict = {0: 1, 1: 0, 2: -1, -1:2}

    def call(self, inputs):
        return tf.convert_to_tensor(self.__concat_result(inputs.numpy()))

