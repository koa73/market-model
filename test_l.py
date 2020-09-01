import tensorflow as tf
import ShaperLayer as c
import numpy as np
import csv
import os


__fileDir = os.path.dirname(os.path.abspath(__file__))
inputDir = __fileDir + '/data/test/tmp/'
filename = inputDir + 'NONE.csv'
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
        my_sum = c.ShaperLayer(2)
        y = my_sum(x)
        raw_data.append(y.numpy())
        if(y.numpy() > 0):
            up +=1
        elif(y.numpy() < 0):
            down +=1
        else:
            none +=1
    f.close()
print('UP : ' + str(up) + ', NONE: ' + str(none) + ', DOWN : '+ str(down) + ', SUMM : '+ str(up+none+down) )
