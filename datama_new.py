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
    __date_array = []
    __data_len = 0
    # Количество столбцов данных выбираемых из файла
    __data_col = 5

    np.random.seed(42)

    __data_mean = np.empty(shape=[0, __data_col])
    __data_std = np.empty(shape=[0, __data_col])

    def __init__(self, tiker, batch_size, control_size):

        self.__tiker = tiker
        self.__batch_size = batch_size
        self.__control_size = control_size
        self.__read_data()
        self.__data_len = len(self.__full_data)

    def get_current_dir(self):
        return self.__fileDir


    def __read_data(self):
        # Формирует массив данных из файла CSV, производит нормализацию
        try:
            with open(self.__fileDir + '/data/' + self.__tiker+'.csv', newline='') as f:
                # remove first string with column name data
                next(f)
                rows = csv.reader(f, delimiter=';', quotechar='|')

                for row in rows:
                    self.__date_array.append(row[2])
                    self.__full_data.append(row[4:])

                self.__get_norma_values()

        except FileNotFoundError:
            print('Error: File "' + self.__tiker + '.csv" not found')

    def __get_data(self, start, end, x_array_norm=True, x_shape_3d=False):
        x_array = []
        y_array = []

        #n_array = self.__norma(np.delete(np.array(self.__full_data[start:end]), (5,), axis=1))
        n_array = np.delete(np.array(self.__full_data[start:end],  dtype="float64"), np.s_[5:], axis=1)
        dn_array = np.array(self.__full_data[start:end],  dtype="float64")

        data_len = n_array.shape[0]

        for ii in range(0, self.__batch_size):
            for i in range(ii, data_len-self.__batch_size+1, self.__batch_size):
                end = i+self.__batch_size-self.__control_size
                if x_array_norm:
                    x_array.append(np.concatenate(self.__norma(n_array[i:end]), axis=None))
                else:
                    x_array.append(np.concatenate(n_array[i:end], axis=None))
                # Удаление лишних данных (<OPEN> High Low <CLOSE><VAL>)
                y_array.append(
                    np.concatenate((np.delete(dn_array[end:end + self.__control_size], np.s_[0, 1, 2, 3, 4, 7, 8], 1)),
                                              axis=None))
        if (x_shape_3d):
            return self.__reshape_x_array(np.array(x_array, dtype="float64")), np.array(y_array, dtype="float64")
        else:
            return np.array(x_array, dtype="float64"), np.array(y_array, dtype="float64")

    def __get_data_(self, start, end, x_array_norm=True, x_shape_3d=False):
        x_array = []
        y_array = []

        n_array = np.delete(np.array(self.__full_data[start:end],  dtype="float64"), np.s_[5:], axis=1)
        dn_array = np.array(self.__full_data[start:end], dtype="float64")

        data_len = n_array.shape[0]

        for i in range(0, data_len-self.__batch_size+1):
            end = i + self.__batch_size-self.__control_size
            if x_array_norm:
                x_array.append(np.concatenate(self.__norma(n_array[i:end]), axis=None))
            else:
                x_array.append(np.concatenate(n_array[i:end], axis=None))
            # Удаление лишних данных (<OPEN> High Low <CLOSE><VAL>)
            y_array.append(
                np.concatenate((np.delete(dn_array[end:end + self.__control_size], np.s_[0, 1, 2, 3, 4, 7, 8], 1)),
                               axis=None))

        if (x_shape_3d):
            return self.__reshape_x_array(np.array(x_array,  dtype="float64")), np.array(y_array,  dtype="float64")
        else:
            return np.array(x_array,  dtype="float64"), np.array(y_array,  dtype="float64")

    def __reshape_x_array(self, data):
        """
        Изменение формы массива X из 2D в 3D
        :return:
        """
        return np.reshape(np.expand_dims(data, axis=1), (data.shape[0], self.__batch_size - self.__control_size,
                                                         self.__data_col))

    def __norma(self, data):
        """
        :param data: Массив данных
        :return: Нормализованный массив data_return, среднее по столбцу data_mean, стандартное отклонение по столбцу data_std
        """
        data_std = data.std(axis=0, dtype=np.float64)  # Определяем стандартное отклонение по каждому столбцу
        data_mean = data.mean(axis=0, dtype=np.float64)  # Вычисляем среднее по каждому столбцу
        data_return = data.astype(np.float64)  # Приведение типов
        data_return -= data_mean  # Вычитаем среднее
        data_return /= data_std  # Делим на отклонение
        return data_return

    def reshapy_y_by_coll(self, y_array, remove_col=1):
        """

        :param y_array:
        :param remove_col: удаляет 1 столбец из массива, по умолчанию <Low> значение
        :return:
        """
        return np.concatenate(np.delete(y_array, (remove_col,), axis=1), axis=None)

    def save_conf(self, model):
        """
        :param model:
        :return: Null
        """
        json_file = open(self.__fileDir + "\models\weights.json", "w")
        json_file.write(model.to_json())
        json_file.close()

    def __get_date_range(self):
        return np.array(self.__date_array[1-self.__batch_size:])
    """
       ****************************************************************************************************************

       ****************************************************************************************************************
       """

    def get_edu_data(self, x_array_3d=False):
        """
        Возвращает массивы данных для обучения сети
        :x_array_3d: - возвращать массмв Х в виде 3D array
        :return:
        """
        data_len = self.__data_len - self.__batch_size * 20
        print('--->> ', data_len)
        return self.__get_data_(0, data_len, x_array_3d)

    def get_test_data(self, x_array_3d=False):
        """
        Возвращает данные не участвовавшие в обучениии модели
        :x_array_3d: - возвращать массмв Х в виде 3D array
        :return:
        """
        data_len = self.__data_len - self.__batch_size * 20
        return self.__get_data_(data_len, None,  x_array_3d)

    def get_test_denorm_data(self, x_array_3d=False):
        """
        Возвращает данные не участвовавшие в обучениии модели
        :x_array_3d: - возвращать массмв Х в виде 3D array
        :return:
        """
        data_len = self.__data_len - self.__batch_size * 20
        return self.__get_data_(data_len, None, False,  x_array_3d)[0]

    def get_predict_data(self, x_array_3d=False):
        """
        :return: X_predict  - массив для предсказания
        """
        #return np.expand_dims(np.concatenate(self.__norma(np.array(self.__full_data[1-self.__batch_size+1:])), axis=None),
         #axis=0)
        return np.expand_dims(np.concatenate(self.__norma(np.delete(np.array(self.__full_data[1-self.__batch_size:],  dtype="float64"), np.s_[5:],
                                     axis=1)), axis=None), axis=0)

    def predict_report(self, predict):

        X_array = np.delete(np.array(self.__full_data[1 - self.__batch_size:],  dtype="float64"), np.s_[5:], axis=1)
        date_array = self.__get_date_range()

        for i in range(len(X_array)):
            print(date_array[i], X_array[i][1])

        print('------------------------------')

        result = X_array[-1][1]*predict/10
        print('> HIGH next day : %f' % result)
