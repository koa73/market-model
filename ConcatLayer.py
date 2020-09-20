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

        tmp = np. array([up[idx], none[idx], down[idx]], dtype='float32')
        max_index_array = np.argmax(tmp, axis=0)

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
            return  vector

        return np.array([0,0,0])

    def __concat_result(self, inputs):

        __array = inputs.numpy()

        vector_up = np.array(__array).astype(np.float32)[0:3]
        vector_none = np.array(__array).astype(np.float32)[3:6]
        vector_down = np.array(__array).astype(np.float32)[6:9]

        max_index_up = self.__get_max_index(vector_up)
        max_index_none = self.__get_max_index(vector_none)
        max_index_down = self.__get_max_index(vector_down)

        calc_value = abs(max_index_none) * (max_index_up + max_index_down + max_index_none)

        vector_up = self.__remove_ex_data(vector_up, max_index_up, calc_value)
        vector_none = self.__remove_ex_data(vector_none, max_index_none, calc_value)
        vector_down = self.__remove_ex_data(vector_down, max_index_down, calc_value)

        if (calc_value == 0):
            return self.__find_best_data(vector_up,vector_none,vector_down, 1)
        elif (calc_value >= 1):
            return self.__find_best_data(vector_up,vector_none,vector_down, 0)
        elif (calc_value <= -1):
            return self.__find_best_data(vector_up,vector_none,vector_down, 2)

    def __wrapper(self, inputs):
        arr = np.zeros([None, 3])
        if (tf.executing_eagerly()):
            for i in range(0, inputs.shape[0]):
                res = self.__concat_result(tf.slice(inputs, [i, 0], [1, inputs.shape[1]])[0]).reshape(1, -1)
                arr = np.concatenate((arr, res), axis=0)
        return tf.convert_to_tensor(arr)
       # else:
            #return tf.slice(inputs, [0, 0], [-1, 3])


    def call(self, inputs, **kwargs):
        return self.__wrapper(inputs)

    def __init__(self):
        super(ConcatLayer, self).__init__()
        self.convert_dict = {0: 1, 1: 0, 2: -1}

    def build(self, input_shape):
        return super(ConcatLayer, self).build(input_shape)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(ConcatLayer, self).get_config()
        return dict(list(base_config.items()))





