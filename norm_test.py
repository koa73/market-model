#!/usr/bin/env python3


import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization


data =  tf.constant(np.array([[7,4,2,6],[1,0,12,10],[2,5,5,2]]))
layer = tf.keras.layers.LayerNormalization(axis=1)
output = layer(data)
print(output)