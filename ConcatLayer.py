import tensorflow as tf
import numpy as np


class ConcatLayer(tf.keras.layers.Layer):

    def get_max_index(self, vector):

        winner = tf.where(vector == tf.math.reduce_max(vector))
        if(tf.equal(tf.shape(winner)[0], 1)):
            return tf.math.add(tf.math.argmax(tf.reverse(vector, [0])), tf.constant(-1, dtype=tf.int64))
        else:
            return tf.constant(0, dtype=tf.int64)

    def find_best_data(self, up, none, down, idx):

        idx = tf.cond(tf.equal(idx, 0), lambda: tf.constant(1, dtype=tf.int32),
                      lambda: tf.cond(tf.equal(idx, 1), lambda: tf.constant(0, dtype=tf.int32),
                                      lambda: tf.constant(2, dtype=tf.int32)))

        offset = tf.math.multiply(tf.math.argmax(tf.concat([tf.slice(up, [idx], [1]), tf.slice(none, [idx], [1]),
                                                            tf.slice(down, [idx], [1])], 0)), tf.constant(3, dtype=tf.int64))

        return tf.slice(tf.concat([up, none, down], 0), [tf.cast(offset, dtype=tf.int32)], [3])


    def remove_ex_data(self, vector, max_idx, calc_value):

        calc_value = tf.cond(tf.equal(calc_value, 0), lambda: calc_value, lambda: tf.cond(
            tf.greater(calc_value, 0), lambda: tf.constant(1, dtype=tf.int64), lambda: tf.constant(-1, dtype=tf.int64)))

        return tf.math.multiply(vector, tf.cond(tf.equal(calc_value, max_idx), lambda: tf.constant(1., dtype=tf.float64)
                                                , lambda: tf.constant(0., dtype=tf.float64)))

    def concat_result(self, vector):

        vector_up = tf.slice(vector, [0], [3])
        vector_none = tf.slice(vector, [3], [3])
        vector_down = tf.slice(vector, [6], [3])

        max_index_up = self.get_max_index(vector_up)
        max_index_none = self.get_max_index(vector_none)
        max_index_down = self.get_max_index(vector_down)

        calc_value = tf.math.multiply(tf.math.abs(max_index_none),
                                      tf.math.add_n([max_index_up, max_index_down, max_index_none]))
        #calc_value = tf.math.add_n([max_index_up, max_index_down, max_index_none])

        vector_up = self.remove_ex_data(vector_up, max_index_up, calc_value)
        vector_none = self.remove_ex_data(vector_none, max_index_none, calc_value)
        vector_down = self.remove_ex_data(vector_down, max_index_down, calc_value)

        return self.find_best_data(vector_up, vector_none, vector_down, calc_value)

    @tf.function(autograph=True)
    def wrapper(self, inputs):
        #_, self.total = tf.while_loop(cond=lambda i, output: tf.less(i, tf.shape(inputs)[0]),
        #                          body=lambda i, output: (i+1, tf.concat([output, (tf.reshape(
        #                              self.concat_result(inputs[i]), [1, 3]))], 0)),
        #                          loop_vars=(tf.constant(0, dtype=tf.int32), self.total)
        #                         )
        for i in tf.range(0, tf.shape(inputs)[0]):
            self.total = tf.concat([self.total, (tf.reshape(self.concat_result(inputs[i]), [1, 3]))], 0)

        return self.total

    def __init__(self, trainable=False, name=None, dtype=tf.float64, dynamic=False, **kwargs):
        self.output_dim = 3
        self.total = tf.Variable((np.empty((0, 3), dtype=np.float64)), shape=[None, 3])
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(ConcatLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.wrapper(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def _add_trackable(self, trackable_object, trainable):
        return super()._add_trackable(trackable_object, trainable)

    def get_config(self):

      base_config = super(ConcatLayer, self).get_config()
      return dict(list(base_config.items()))
