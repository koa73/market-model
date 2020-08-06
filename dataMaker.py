#!/usr/bin/env python3

import csv
import os
import re
from decimal import Decimal as D, ROUND_DOWN
from datetime import datetime
import numpy as np


class DataMaker:

    __fileDir = os.path.dirname(os.path.abspath(__file__))
    __tikets = []
    __accuracy = '0.00001'
    __max_border = 0.05
    __min_border = -0.05

    def __init__(self, batch_size=3):
        self.__batch_size = batch_size

    # Вычисление изменения текущего значения относительно базового
    def __change_percent(self, base, curr):

        try:
            return float(D((float(curr) - float(base))/float(base)).quantize(D(self.__accuracy), rounding=ROUND_DOWN))
        except ZeroDivisionError:
            return float(1)

    # Запись данных в СSV файл
    def __append_to_file(self, name, data, subdir):

        filename = self.__fileDir + '/data/' + subdir + 'train_' + name + '.csv'
        if os.path.isfile(filename):
            os.remove(filename)

        with open(filename, 'a', newline = '') as csv_out_file:
            output = csv.writer(csv_out_file, delimiter=';')

            output.writerow(['Date', 'Open', 'Low', 'High', 'Close', 'Adj' 'Close', 'Volume',
                            'DAY', 'C1', "Low'", "High'", "Close/Current'", "C0", "Volume'"])

            for line in data:
                output.writerow(line)

        csv_out_file.close()
