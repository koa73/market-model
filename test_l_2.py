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
    raw_data = []
    for row in rows:
        # Превращает вектор [1,9] в тензор
        x = tf.constant(list(np.float_(row[0:9])), dtype='float64')
        # Инициирует слой
        separator = c.ConcatLayer()
        # Получаем выход вектор [1,3]
        y = separator(x)
        # Классифицируем ответ
        calc_val = get_max_index(y)
        if(calc_val == 0):
            none +=1
        elif(calc_val == 1):
            up +=1
        else:
            down +=1
    f.close()
print('UP : ' + str(up) + ', NONE: ' + str(none) + ', DOWN : '+ str(down) + ', SUMM : '+ str(up+none+down) )

