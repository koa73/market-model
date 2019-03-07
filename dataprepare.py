import csv
import os
import numpy as np


class DataPrepare:

    __tiker = ""
    __fileDir = os.path.dirname(os.path.abspath(__file__))
    # размер пакета случайной выборки из массива
    __batch_size = 0
    # количество строк проверочных данных в выборке, должно быть меньше или равно min-3 значения batch_size_range
    __control_size = 1
    # Номер первой колонки для выборки из файла данных
    __col_start = 4
    # Количество столбцов данных выбираемых из файла
    __data_col = 4

    __mean = 0
    __std = 0

    __full_data = []

    def __init__(self, tiker, batch_size, control_size):

        self.__tiker = tiker
        self.__batch_size = batch_size
        self.__control_size = control_size
        self.__read_data()
        self.__data_len = len(self.__full_data)

    @staticmethod
    def __convert_to_float(val):
        # Производит приведение данных к формату float
        if not isinstance(val, float):
            return float(val)

    def __read_data(self):

        __date_array = []
        # Формирует массив данных из файла CSV, производит нормализацию
        try:
            with open(self.__fileDir + '/data/' + self.__tiker+'.csv', newline='') as f:
                # remove first string with column name data
                next(f)
                rows = csv.reader(f, delimiter=';', quotechar='|')

                for row in rows:
                    __date_array.append(row[2])
                    new_row = []

                    for elem in row[self.__col_start:self.__col_start+self.__data_col]:
                        new_row.append(self.__convert_to_float(elem))
                    self.__full_data.append(new_row)

        except FileNotFoundError:
            print('Error: File "' + self.__tiker + '.csv" not found')

    def __get_data(self, start, end):
        x_array = []
        y_array = []

        n_array = np.array(self.__full_data[start:end])
        dn_array = np.array(self.__full_data[start:end])

        data_len = n_array.shape[0]

        for ii in range(0, self.__batch_size):
            for i in range(ii, data_len-self.__batch_size+1, self.__batch_size):
                end = i+self.__batch_size-self.__control_size
                x_array.append(np.concatenate(n_array[i:end], axis=None))
                # Удаление лишних данных (<OPEN> High Low <CLOSE><VAL>)
                y_array.append(np.concatenate((np.delete(dn_array[end:end + self.__control_size], np.s_[0, 3, 4], 1)),
                                              axis=None))

        return np.array(x_array), np.array(y_array)

    def __get_data_(self, start, end):
        x_array = []
        y_array = []

        n_array = np.array(self.__full_data[start:end], dtype="float64")
        data_len = n_array.shape[0]

        for i in range(0, data_len-self.__batch_size+1):
            end = self.__batch_size-self.__control_size
            n_array_slice = n_array[i:i + self.__batch_size]
            x_array.append(np.concatenate(self.__norma_x(n_array_slice[:end]), axis=None))
            # Удаление лишних данных (<OPEN> High Low <CLOSE><VAL>)
            y_array.append(
                np.concatenate((np.delete(self.__norma_y(n_array_slice[end:]), np.s_[0, 3, 4], 1)),
                               axis=None))

        return np.array(x_array,  dtype="float64"),  np.array(y_array,  dtype="float64")

    def __reshape_x_array(self, data):
        """
        Изменение формы массива X из 2D в 3D
        :return:
        """
        return np.reshape(np.expand_dims(data, axis=1), (data.shape[0], int(data[0].shape[0]/self.__data_col),
                                                         self.__data_col))

    def __norma_x(self, data):
        """
        :param data: Массив данных
        :return: Нормализованный массив data_return, среднее по столбцу data_mean, стандартное отклонение по столбцу data_std
        """
        self.__std = data.std(axis=0, dtype=np.float64)  # Определяем стандартное отклонение по каждому столбцу
        self.__mean = data.mean(axis=0, dtype=np.float64)  # Вычисляем среднее по каждому столбцу
        data_return = data.astype(np.float64)  # Приведение типов
        data_return -= self.__mean  # Вычитаем среднее
        data_return /= self.__std  # Делим на отклонение
        return data_return

    def __norma_y(self, data):
        """
        :param data: Массив данных
        :return: Нормализованный массив data_return, среднее по столбцу mean, стандартное отклонение по столбцу std
        """
        data_return = data.astype(np.float64)  # Приведение типов
        data_return -= self.__mean  # Вычитаем среднее
        data_return /= self.__std  # Делим на отклонение
        return data_return


    """
        ****************************************************************************************************************

        ****************************************************************************************************************
    """
    def get_edu_data(self, x_shape_3d=False):
        """
        Возвращает массивы данных для обучения сети
        :return:
        """
        x_i =[]

        data_len = self.__data_len - self.__batch_size * 0
        x, y = self.__get_data_(0, data_len)
        if (x_shape_3d):
            x_i.append(self.__reshape_x_array(x))
        else:
            x_i.append(x)

        for i in range(self.__data_col, (self.__batch_size-self.__control_size-1)*self.__data_col, self.__data_col):
            if (x_shape_3d):
                x_i.append(self.__reshape_x_array(np.delete(x, np.s_[:i:], 1)))
            else:
                x_i.append(np.delete(x, np.s_[:i:], 1))

        return x_i, y
