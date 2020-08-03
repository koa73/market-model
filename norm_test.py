import keras
import numpy as np
import tensorflow as tf


data = np.array([[7,4,2],[1,0,12],[5,5,2]])
print(data)
layer = tf.keras.layers.LayerNormalization(axis=1)
output = layer(data)
print(output)