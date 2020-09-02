import tensorflow as tf
import ConcatLayer as c
import numpy as np
import csv
import os

convert_dict = {0: 1, 1: 0, 2: -1}

def get_max_index(vector):

    winner = np.argwhere(vector == np.amax(vector))
    if (winner.size > 1):
        return 0
    else:
        return convert_dict[winner[0][0]]

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
        calc_val = get_max_index(y)
        input (calc_val)
    f.close()

