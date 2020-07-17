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
        self.check_dictionary('/data/output/')


    def __get_tickers(self, subdir):
        return [re.findall(r'.*_(\w+)\.\w{3}', f.name)[0] for f in os.scandir(self.__fileDir+ subdir) if f.is_file()]

    def __read_data_new(self):

        __ticker = 'Unknown file'
        dirOutput = '/data/stock/'
        try:

            for __ticker in self.__get_tickers(dirOutput):

                raw_data = []
                with open(self.__fileDir + dirOutput + 'train_' + __ticker + '.csv', newline='') as f:

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

                X_array_ticker, Y_array_tiker = self.__calculate_col_values(3, raw_data)
                self.__append_to_file(__ticker, raw_data)

        except FileNotFoundError:

            print('Error: File "' + __ticker + '.csv" not found')

    # Check dictionary
    def check_dictionary(self, dirname):

        __ticker = 'Unknown file'
        __d = {}
        count = 0

        print("-------- Dictionary check  ---->> " )
        try:
            for __ticker in self.__get_tickers(dirname):

                raw_data = []
                with open(self.__fileDir + dirname + 'train_' + __ticker + '.csv', newline='') as f:

                    print("---->> " + str(__ticker))

                    next(f)
                    rows = csv.reader(f, delimiter=';', quotechar='|')

                    for row in rows:
                        raw_data.append(row)

                    X_array, Y_array = self.__calculate_col_values(3, raw_data)

                    count += X_array.shape[0]
                    for i in range (0, X_array.shape[0]):

                        key = ";".join(np.array(X_array[i]).flatten().astype(str))

                        if key in __d:
                            __d[key] = __d[key] + 1
                        else:
                            __d[key] = 1
                    print( "Current count :"+str(count))

                f.close()

            print("Common rows : " + str(count) + " dictionary size : " + str(len(__d)))

            i = 0
            if (len(__d) < count):

                for key, value in sorted(__d.items(), key=lambda x: x[1]):

                    if (value > 1):
                        i +=1
                        print("Key :" + str(key) + ", value :" + str(value))

            print("There is dubled " + str(i))

        except FileNotFoundError:
            print('Error: File "' + __ticker + '.csv" not found')



    # Write data to output csv file
    def __append_to_file(self, name, data):

        filename = self.__fileDir + '/data/output/train_' + name + '.csv'
        if os.path.isfile(filename):
            os.remove(filename)

        with open(self.__fileDir + '/data/output/train_'+ name +'.csv', 'a', newline = '') as csv_out_file:
            output = csv.writer(csv_out_file, delimiter=';')

            output.writerow(['Date', 'Open', 'Low', 'High', 'Close', 'Adj' 'Close', 'Volume',
                            'DAY', 'Carrier', "Low'", "High'", "Close/Current'"])

            for line in data:
                output.writerow(line)

        csv_out_file.close()


    def __change_percent(self, current, next):

        return float(D((float(next) - float(current))/float(current)*100).quantize(D('0.01'), rounding=ROUND_DOWN))

    def __calculate_col_values(self, range_size, raw_data):

        n_array = (np.delete(np.array(raw_data), (0, 1, 5, 6, 7), axis=1)).astype(np.float64)
        data_len = n_array.shape[0]

        Y_array = []
        X_array = []

        for i in range(0, data_len - 2 * range_size + 1):
            end = i + range_size

            # Find max & min in feature slice period
            f_max = np.max(n_array[i + range_size:end + range_size], axis=0)[1]
            f_min = np.min(n_array[i + range_size:end + range_size], axis=0)[0]


            # Find Low & High change in feature slice period
            f_ch_percent_low = self.__change_percent(str(n_array[i:end][-1][2]), f_min)
            f_ch_percent_high = self.__change_percent(str(n_array[i:end][-1][2]), f_max)

            # Remove abs values from array
            X_row = np.delete(n_array[i:end], np.s_[0, 1, 2], 1)
            Y_row = np.array([f_ch_percent_low,f_ch_percent_high])

            Y_array.append(Y_row)
            X_array.append(X_row)

        return np.array(X_array), np.array(Y_array)


    # Convert date to day of year
    def __day_of_year(self, date_str):
        return datetime.strptime(date_str, '%Y-%m-%d').date().timetuple().tm_yday
