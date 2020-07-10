import csv
import os
import re
from decimal import Decimal as D, ROUND_DOWN
import numpy as np

class DataMiner:

    __fileDir = os.path.dirname(os.path.abspath(__file__))
    __tikets =[]

    def __init__(self, tiker, batch_size):

        self.__tiker = tiker
        self.__read_data()
        self.__batch_size = batch_size

    def __get_tickers(self):
        return [re.findall(r'.*_(\w+)\.\w{3}', f.name)[0] for f in os.scandir(self.__fileDir+ '/data/stock/') if f.is_file()]


    def __change_percent(self, current, next):
        from decimal import Decimal as D, ROUND_DOWN
        return D((float(next) - float(current))/float(current)*100).quantize(D('0.001'), rounding=ROUND_DOWN)


    def __read_data(self):

        try:

            raw_data = []

            for __ticker in self.__get_tickers():

                with open(self.__fileDir + '/data/stock/train_' + __ticker+'.csv', newline='') as f:

                    next(f)
                    rows = csv.reader(f, delimiter=',', quotechar='|')
                    row = next(rows)

                    while True:
                        try:
                            next_row = next(rows)
                            #print(__ticker + "\t " + row[0] + "\tLow : " + row[2] + "\tHigh : " + row[3] + "\t "
                            #      + "\tLow change % :" + str(self.__change_percent(row[2], next_row[2]))
                            #      + "\tHigh change % :" + str(self.__change_percent(row[3], next_row[3])))

                            raw_data.append([__ticker, row[0], float(row[2]), float(row[3]),
                                             self.__change_percent(row[2], next_row[2]),
                                             self.__change_percent(row[3], next_row[3])])

                            row = next_row

                        except StopIteration:
                            break

                f.close()

                print(raw_data)
                input("Press any key ...")

        except FileNotFoundError:
            print('Error: File "' + __ticker + '.csv" not found')