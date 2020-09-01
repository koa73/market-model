import tensorflow as tf
import ShaperLayer as c
import numpy as np
import csv
import os


__fileDir = os.path.dirname(os.path.abspath(__file__))
inputDir = __fileDir + '/data/test/tmp/'
filename = inputDir + 'UP.csv'


with open(filename, newline='') as f:
    rows = csv.reader(f, delimiter=';', quotechar='|')
    i=0
    for row in rows:

        x = tf.constant(list(np.float_(row[0:9])))
        my_sum = c.ShaperLayer(2)
        y = my_sum(x)
        #input(y.numpy())
    f.close()