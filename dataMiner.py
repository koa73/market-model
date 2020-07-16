import csv
import os
import re
from decimal import Decimal as D, ROUND_DOWN
from datetime import datetime
import numpy as np

class DataMiner:

    __fileDir = os.path.dirname(os.path.abspath(__file__))
    __tikets =[]

    def __init__(self, tiker, batch_size):

        self.__tiker = tiker
        self.__batch_size = batch_size
        self.__read_data_new()


    def __get_tickers(self):
        return [re.findall(r'.*_(\w+)\.\w{3}', f.name)[0] for f in os.scandir(self.__fileDir+ '/data/stock/') if f.is_file()]


    def __read_data_new(self):

        try:

            raw_data = []

            for __ticker in self.__get_tickers():

                self.__init_output_file(__ticker)

                with open(self.__fileDir + '/data/stock/train_' + __ticker + '.csv', newline='') as f:

                    next(f)
                    rows = csv.reader(f, delimiter=',', quotechar='|')
                    row = next(rows)
                    print("---- " + __ticker + " ----------")

                    while True:
                        try:

                            next_row = next(rows)

                            # Find carrier as a change of open in percentages
                            carrier = self.__change_percent(row[1], next_row[1])
                            row = next_row

                            # Find day of year
                            day_of_year = self.__day_of_year(row[0])
                            row.append(day_of_year)
                            row.append(carrier)

                            low_ = self.__change_percent(row[1], row[2])
                            row.append(low_)

                            high_ = self.__change_percent(row[1], row[3])
                            row.append(high_)

                            close_current_ = self.__change_percent(row[1], row[4])
                            row.append(close_current_)

                            raw_data.append(row)

                        except StopIteration:
                            break

                f.close()
                input("Press any key ... ")
                self.__append_to_file(__ticker, raw_data)

        except FileNotFoundError:

            print('Error: File "' + __ticker + '.csv" not found')


    # Write data to output csv file
    def __append_to_file(self, name, data):

        with open(self.__fileDir + '/data/output/train_'+ name +'.csv', 'a', newline = '') as csv_out_file:
            output = csv.writer(csv_out_file, delimiter=';')
            for line in data:
                output.writerow(line)
        csv_out_file.close()

    def __init_output_file(self, name):

        filename = self.__fileDir + '/data/output/train_' + name + '.csv'
        if os.path.isfile(filename):
            os.remove(filename)
            with open(self.__fileDir + '/data/output/train_' + name + '.csv', 'w', newline='') as csv_out_file:
                output = csv.writer(csv_out_file, delimiter=';')
                output.writerow(['Date','Open','Low','High','Close','Adj' 'Close','Volume',
                                                     'DAY', 'Carrier', "Low'", "High'", "Close/Current'"])
            csv_out_file.close()


    def __change_percent(self, current, next):

        return float(D((float(next) - float(current))/float(current)*100).quantize(D('0.001'), rounding=ROUND_DOWN))


    def __calculate_col_values(self, raw_data):

        n_array = (np.delete(np.array(raw_data), (0, 1, 7), axis=1)).astype(np.float64)
        data_len = n_array.shape[0]

        X_array = []
        Y_array = []

        for i in range(0, data_len - 2 * self.__batch_size + 1):
            end = i + self.__batch_size

            # Find max & min in feature slice period
            f_max = np.max(n_array[i + self.__batch_size:end + self.__batch_size], axis=0)[1]
            f_min = np.min(n_array[i + self.__batch_size:end + self.__batch_size], axis=0)[0]

            # Find Low & High change in feature slice period
            f_ch_percent_low = self.__change_percent(str(n_array[i:end][-1][0]), f_min)
            f_ch_percent_high = self.__change_percent(str(n_array[i:end][-1][1]), f_max)

            # Remove abs values from array
            X_row = np.delete(n_array[i:end], np.s_[0, 1], 1)
            Y_row = np.array([f_ch_percent_low,f_ch_percent_high])

            X_array.append(X_row)
            Y_array.append(Y_row)

        return np.array(X_array), np.array(Y_array)


    def __day_of_year(self, date_str):
        return datetime.strptime(date_str, '%Y-%m-%d').date().timetuple().tm_yday


    # Old reader
    def __read_data(self):

        try:

            X_array = []
            Y_array = []
            for __ticker in self.__get_tickers():

                raw_data = []
                x_array = []
                y_array = []

                print ("------------------------ "+__ticker +" --------------------\n")

                with open(self.__fileDir + '/data/stock/train_' + __ticker+'.csv', newline='') as f:

                    next(f)
                    rows = csv.reader(f, delimiter=',', quotechar='|')
                    row = next(rows)

                    while True:
                        try:
                            next_row = next(rows)

                            change_low = self.__change_percent(row[2], next_row[2])
                            change_high = self.__change_percent(row[3], next_row[3])
                            change_close = self.__change_percent(next_row[1], next_row[4])
                            row = next_row
                            raw_data.append([__ticker, row[0], float(row[2]), float(row[3]),
                                             change_low, change_high, change_close, self.__day_of_year(row[0])])


                        except StopIteration:
                            break


                x_array, y_array = self.__calculate_col_values(raw_data)

                X_array.append(x_array)
                Y_array.append(y_array)


                f.close()
                self.__calculate_col_values(raw_data)


        except FileNotFoundError:
            print('Error: File "' + __ticker + '.csv" not found')