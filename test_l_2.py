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
filename = inputDir + 'DOWN_1.csv'
up = 0
down = 0
none = 0

with open(filename, newline='') as f:

    rows = csv.reader(f, delimiter=';', quotechar='|')
    raw_data = []
    i = 0
    for row in rows:
        i += 1
        raw_data.append(list(np.float_(row[0:9])))
        if (i == 10):
            # Превращает вектор [1,9] в тензор
            x = tf.constant(raw_data)

            #x = tf.constant([[0.01,0.06,0.06, 0.14,0.7,0.16, 0.2,0.11,0.69],
            #                 [0.49,0.4,0.12, 0.85,0.99,0.10, 0.1,0.21,0.84]])

            # Инициирует слой
            separator = c.ConcatLayer()
            # Получаем выход вектор [1,3]
            y = separator(x)
            # Классифицируем ответ
            for j in range(0, y.shape[0]):
               calc_val = get_max_index(y[j].numpy())
               if (calc_val == 0):
                   none += 1
               elif (calc_val == 1):
                   up += 1
               else:
                   down += 1
            raw_data = []
            i = 0
    if (len(raw_data)>0):
        # Превращает вектор [1,9] в тензор
        x = tf.constant(raw_data)
        # Инициирует слой
        separator = c.ConcatLayer()
        # Получаем выход вектор [1,3]
        y = separator(x)
        # Классифицируем ответ
        for j in range(0, y.shape[0]):
            calc_val = get_max_index(y[j].numpy())
            if (calc_val == 0):
                none += 1
            elif (calc_val == 1):
                up += 1
            else:
                down += 1
        raw_data = []

    f.close()
print('UP : ' + str(up) + ', NONE: ' + str(none) + ', DOWN : '+ str(down) + ', SUMM : '+ str(up+none+down) )

