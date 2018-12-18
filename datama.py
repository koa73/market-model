import csv
import os
import numpy as np
import datetime
import re


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

    @staticmethod
    def __convert_to_float(val):
        # Производит приведение данных к формату float
        if not isinstance(val, float):
            return float(val)

    @staticmethod
    def __rebuild_date(str):
        m = re.findall("(\d{4})(\d{2})(\d{2})", str)[0]
        return m[2]+'.'+m[1]+'.'+m[0]

    def __read_data(self):
        # Формирует массив данных из файла CSV, производит нормализацию
        try:
            with open(self.__fileDir + '/data/' + self.__tiker+'.csv', newline='') as f:
                # remove first string with column name data
                next(f)
                rows = csv.reader(f, delimiter=';', quotechar='|')

                for row in rows:

                    self.__date_array.append(self.__rebuild_date(row[2]))
                    new_row = []

                    for elem in row[4:9]:
                        new_row.append(self.__convert_to_float(elem))
                    self.__full_data.append(new_row)

                self.__get_norma_values()

        except FileNotFoundError:
            print('Error: File "' + self.__tiker + '.csv" not found')

    def __get_data(self, start, end, x_shape_3d=False):
        x_array = []
        y_array = []

        n_array = self.__norma(np.array(self.__full_data[start:end]))
        #n_array = np.array(self.__full_data[start:end])

        data_len = n_array.shape[0]

        for ii in range(0, self.__batch_size):
            for i in range(ii, data_len-self.__batch_size+1, self.__batch_size):
                end = i+self.__batch_size-self.__control_size
                x_array.append(np.concatenate(n_array[i:end], axis=None))
                # Удаление лишних данных (<OPEN> High Low <CLOSE><VAL>)
                y_array.append(np.concatenate((np.delete(n_array[end:end + self.__control_size], np.s_[0, 3, 4], 1)),
                                              axis=None))

        if (x_shape_3d):
            #return self.__reshape_x_array(np.array(x_array)), self.__denorm_y_array(np.array(y_array))
            return self.__reshape_x_array(np.array(x_array)), np.array(y_array)
        else:
            #return np.array(x_array), self.__denorm_y_array(np.array(y_array))
            return np.array(x_array), np.array(y_array)

    def __get_data_(self, start, end, x_shape_3d=False):
        x_array = []
        y_array = []

        #n_array = self.__norma(np.array(self.__full_data[start:end]))
        n_array = np.array(self.__full_data[start:end])

        data_len = n_array.shape[0]

        for i in range(0, data_len-self.__batch_size+1):
            end = i + self.__batch_size-self.__control_size
            x_array.append(np.concatenate(n_array[i:end], axis=None))
            # Удаление лишних данных (<OPEN> High Low <CLOSE><VAL>)
            y_array.append(np.concatenate((np.delete(n_array[end:end + self.__control_size], np.s_[0, 3, 4], 1)),
                                          axis=None))


        if (x_shape_3d):
            #return self.__reshape_x_array(np.array(x_array)), self.__denorm_y_array(np.array(y_array))
            return self.__reshape_x_array(np.array(x_array)), np.array(y_array)
        else:
            #return np.array(x_array), self.__denorm_y_array(np.array(y_array))
            return np.array(x_array), np.array(y_array)

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
        data_return = data.astype(np.float64)  # Приведение типов
        data_return -= self.__data_mean  # Вычитаем среднее
        data_return /= self.__data_std  # Делим на отклонение
        return data_return

    def __get_norma_values(self):
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
        json_file = open(self.__fileDir + "\models\last_" + str(ts) + ".json", "w")
        json_file.write(model.to_json())
        json_file.close()
        model.save_weights(self.__fileDir + "\models\last_" + str(ts) + ".h5")

    def denorm_y_array(self, data):
        """
        Денормализация Y массива
        :param data:
        :return:
        """
        #data *= np.tile(self.__data_std[1:3], self.__control_size)
        #data += np.tile(self.__data_mean[1:3], self.__control_size)
        data *= np.tile(self.__data_std[1:2], self.__control_size)
        data += np.tile(self.__data_mean[1:2], self.__control_size)
        return data

    def denorm_x_array(self, data):
        """
        Денормализация X массива
        :param data:
        :return:
        """
        factor = int(data.shape[-1]/self.__data_col)
        data *= np.tile(self.__data_std, factor)
        data += np.tile(self.__data_mean, factor)
        return data

    def reshapy_y_by_coll(self, y_array, remove_col=1):
        """

        :param y_array:
        :param remove_col: удаляет 1 столбец из массива, по умолчанию <Low> значение
        :return:
        """
        return np.concatenate(np.delete(y_array, (remove_col,), axis=1), axis=None)

    def denorm_y(self, data, i=0):
        """

        :param data:
        :param i: выбирает 1 из параметров нормализации из массива
        :return:
        """
        data *= np.tile(self.__data_std[1:3][i], data.shape[0])
        data += np.tile(self.__data_mean[1:3][i], data.shape[0])
        return data

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
        data_len = self.__data_len - self.__batch_size * 2
        print('--->> ',data_len)
        return self.__get_data(0, data_len, x_array_3d)

    def get_test_data(self, x_array_3d=False):
        """
        Возвращает данные не участвовавшие в обучениии модели
        :x_array_3d: - возвращать массмв Х в виде 3D array
        :return:
        """
        data_len = self.__data_len - self.__batch_size * 2
        return self.__get_data(data_len, None,  x_array_3d)

    def predict_report(self, y_test, predict):
        """
        Временная функция вывода результатов
        :param y_test:
        :param predict:
        :return:
        """
        print("----------------------------- Test -----------------------------------------------")
        for i in range(len(y_test)):
            print(predict[i], y_test[i], "\t",
                  [y_test[i][0] - predict[i][0], predict[i][1] - y_test[i][1]])



