import tensorflow as tf
import ConcatLayer as c
import numpy as np
import csv
import os


__fileDir = os.path.dirname(os.path.abspath(__file__))
inputDir = __fileDir + '/data/test/tmp/'
filename = inputDir + 'UP.csv'
up = 0
down = 0
none = 0

with open(filename, newline='') as f:
    rows = csv.reader(f, delimiter=';', quotechar='|')
    i=0
    raw_data = []
    for row in rows:
        i +=1
        x = tf.constant(list(np.float_(row[0:9])), dtype='float32')
        my_sum = c.ConcatLayer()
        y = my_sum(x)
        input(y.numpy())
    f.close()
