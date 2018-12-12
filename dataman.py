import csv
import os
import numpy as np
import datetime


class DataManager:

    __filename = ""
    __fileDir = os.path.dirname(os.path.realpath('__file__'))
    # размер пакета случайной выборки из массива  >=4
    __batch_size = 0
    # количество строк проверочных данных в выборке, должно быть меньше или равно min-3 значения batch_size_range
    __control_size = 1

    __full_data = []
    __data_len = 0

    np.random.seed(42)

    __data_mean = np.empty(shape=[0, 5])
    __data_std = np.empty(shape=[0, 5])

    def __init__(self, filename, batch_size, control_size):

        self.__filename = filename
        self.__batch_size = batch_size
        self.__control_size = control_size
        self.__read_data()
        self.__data_len = len(self.__full_data)

    def get_current_dir(self):
        return self.__fileDir

    @staticmethod
    def __convert_to_float(val):
        # Производит нормализацию входных данных и приведению их формату float
        if not isinstance(val, float):
            return float(val)

    def __read_data(self):
        # Формирует массив данных из файла CSV, производит нормализацию

        try:
            with open(self.__fileDir + '/data/' + self.__filename, newline='') as f:
                # remove first string with column name data
                next(f)
                rows = csv.reader(f, delimiter=';', quotechar='|')

                for row in rows:
                    new_row = []
                    for elem in row[4:9]:
                        new_row.append(self.__convert_to_float(elem))
                    self.__full_data.append(new_row)

        except FileNotFoundError:
            print('Error: File "' + self.__filename + '" not found')

    def get_data_len(self):
        return self.__data_len

    def get_variations_num(self):
        var_num = 0
        for i in range(1, self.__data_len+1, 1):
            var_num += i//self.__batch_size
        return var_num

    def get_edu_data(self):

        X_array = []
        y_array = []

        n_array = self.__norma(np.array(self.__full_data))
        self.__full_data = []

        for i in range(self.__data_len - self.__batch_size + 1):
            for j in range(i, self.__data_len, self.__batch_size):
                end = j + self.__batch_size
                if end > self.__data_len:
                    break

                X_array.append(n_array[j:end-self.__control_size])
                # Выборка HIGH, LOW
                y_array.append(n_array[end - self.__control_size:end][0][1:3])
        return np.array(X_array), np.array(y_array)

    def __norma(self, data):
        """
        :param data: Массив данных
        :return: Нормализованный массив data_return, среднее по столбцу data_mean, стандартное отклонение по столбцу data_std
        """
        data_return = data.astype(np.float64)  # Приведение типов
        self.__get_normalize_values(data) # Считываем параметры нормализации из файла
        data_return -= self.__data_mean  # Вычитаем среднее
        data_return /= self.__data_std  # Делим на отклонение
        return data_return

    def de_norma(self, data):
        """
        :param data: элемент массива/массив
        :param mean: среднее по столбцу
        :param std: стандартное отклонение по столбцу
        :return: нормализованные данные
        """
        if data.shape[1] == 2:
            data *= self.__data_std[1:3]
            data += self.__data_mean[1:3]
        else:
            data *= self.__data_std  # Умножаем на стандартное отклонение для 0 столбца
            data += self.__data_mean  # Прибавляем среднее для 0 столбца

        return data

    def __remove_extra_columns(self, data):
        data_return = []
        for i in range(self.__control_size):
            data_return.append(data[i][1:3])
        return np.array(data_return)

    def save(self, model):
        now = datetime.datetime.now()
        ts = now.strftime("%d-%m-%Y_%H_%M")
        json_file = open(self.__fileDir + "/models/last_" + str(ts) + ".json", "w")
        json_file.write(model.to_json())
        json_file.close()
        model.save_weights(self.__fileDir + "/models/last_" + str(ts) + ".h5")
        self.__save_normalize_values()

    def __save_normalize_values(self):
        """
        Записываем данные нормализации модели в файл
        :return: Null
        """
        np.save(self.__fileDir + "/data/"+self.__filename[:-4]+"_std.npy", self.__data_std)
        np.save(self.__fileDir + "/data/"+self.__filename[:-4]+"_mean.npy", self.__data_mean)

    def __get_normalize_values(self, data):
        """
                :param data: Массив данных
                :return: Null
                Считывает или вычисляет данные нормализации
        """
        std_file = self.__fileDir + "/data/"+self.__filename[:-4]+"_std.npy"
        mean_file = self.__fileDir + "/data/"+self.__filename[:-4]+"_mean.npy"

        if os.path.exists(std_file):
            self.__data_std = np.load(std_file)
        else:
            self.__data_std = data.std(axis=0)  # Определяем стандартное отклонение по каждому столбцу

        if os.path.exists(mean_file):
            self.__data_mean = np.load(mean_file)
        else:
            self.__data_mean = data.mean(axis=0)  # Вычисляем среднее по каждому столбцу
