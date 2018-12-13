import csv
import os
import numpy as np
import datetime


class DataManager:

    __tiker = ""
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

    def __init__(self, tiker, batch_size, control_size):

        self.__tiker = tiker
        self.__batch_size = batch_size
        self.__control_size = control_size
        self.__read_data()
        self.__data_len = len(self.__full_data)

    def get_current_dir(self):
        return self.__fileDir

    @staticmethod
    def __convert_to_float(val):
        # Производит приведение данных к формату float
        if not isinstance(val, float):
            return float(val)

    def __read_data(self):
        # Формирует массив данных из файла CSV, производит нормализацию
        try:
            with open(self.__fileDir + '/data/' + self.__tiker+'.csv', newline='') as f:
                # remove first string with column name data
                next(f)
                rows = csv.reader(f, delimiter=';', quotechar='|')

                for row in rows:
                    new_row = []
                    for elem in row[4:9]:
                        new_row.append(self.__convert_to_float(elem))
                    self.__full_data.append(new_row)

                self.__get_normalize_values()

        except FileNotFoundError:
            print('Error: File "' + self.__tiker + '.csv" not found')

    def __get_data(self, start, end):
        """
        Формирование одномерного массива для обучения и проверки обучения
        :return:
        """
        x_array = []
        y_array = []

        n_array = self.__norma(np.array(self.__full_data[start:end]))
        #n_array = np.array(self.__full_data[start:end])
        data_len = n_array.shape[0]

        for i in range(data_len - self.__batch_size + 1):
            for j in range(i, data_len, self.__batch_size):
                end = j + self.__batch_size
                if end > data_len:
                    break
                x_array.append(np.concatenate(n_array[j:end - self.__control_size], axis=None))
                # Удаление лишних данных (<OPEN> High Low <CLOSE><VAL>)
                y_array.append(np.concatenate((np.delete(n_array[end - self.__control_size:end], np.s_[0, 3, 4], 1)), axis=None))

        return np.array(x_array), np.array(y_array)

    def __norma(self, data):
        """
        :param data: Массив данных
        :return: Нормализованный массив data_return, среднее по столбцу data_mean, стандартное отклонение по столбцу data_std
        """
        data_return = data.astype(np.float64)  # Приведение типов
        data_return -= self.__data_mean  # Вычитаем среднее
        data_return /= self.__data_std  # Делим на отклонение
        return data_return

    def __get_normalize_values(self):
        """
        Стандартного отклонения и средних по каждому столбщу по обучающим данным
        :return: Null
        """
        end = self.__data_len - self.__batch_size * 2
        n_array = np.array(self.__full_data[0:end])
        self.__data_std = n_array.std(axis=0)  # Определяем стандартное отклонение по каждому столбцу
        self.__data_mean = n_array.mean(axis=0)  # Вычисляем среднее по каждому столбцу

    def save(self, model):
        """
        :param model:
        :return: Null
        """
        now = datetime.datetime.now()
        ts = now.strftime("%d-%m-%Y_%H_%M")
        json_file = open(self.__fileDir + "/models/last_" + str(ts) + ".json", "w")
        json_file.write(model.to_json())
        json_file.close()
        model.save_weights(self.__fileDir + "/models/last_" + str(ts) + ".h5")

    def denorm_y_array(self, data):
        """
        Денормализация Y массива
        :param data:
        :return:
        """
        data *= np.tile(self.__data_std[1:3], self.__control_size)
        data += np.tile(self.__data_mean[1:3], self.__control_size)
        return data

    def denorm_x_array(self, data):
        """
        Денормализация Y массива
        :param data:
        :return:
        """
        data *= self.__data_std
        data += self.__data_mean
        return data

    """
    ****************************************************************************************************************

    ****************************************************************************************************************
    """
    def get_edu_data(self):
        """
        Возвращает массивы данных для обучения сети
        :return:
        """
        data_len = self.__data_len - self.__batch_size * 2
        return self.__get_data(0, data_len)

    def get_test_data(self):
        """
        Возвращает данные участвовавшие в обучении для проверки модели
        :return:
        """
        data_len = self.__data_len - self.__batch_size * 2
        return self.__get_data(data_len - self.__batch_size * 2, data_len)

    def get_verify_data(self):
        """
        Возвращает данные не участвовавшие в обучениии модели
        :return:
        """
        data_len = self.__data_len - self.__batch_size * 2
        return self.__get_data(data_len, None)




