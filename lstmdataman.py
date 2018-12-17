import pandas as pd
import numpy as np
import os
import datetime

def loaddata(filename, separator):
    # Читаем файл
    main_ticker_data = pd.read_csv(filename, sep=separator)
    # Берем только нужные поля
    main_ticker_data = main_ticker_data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
    #main_ticker_data = main_ticker_data[['<OPEN_TAR>', '<HIGH_TAR>', '<LOW_TAR>', '<CLOSE_TAR>', '<VOL_TAR>', '<OPEN_DEP1>', '<HIGH_DEP1>', '<LOW_DEP1>', '<CLOSE_DEP1>']]
    #main_ticker_data = main_ticker_data[['<OPEN_TAR>', '<HIGH_TAR>', '<LOW_TAR>', '<CLOSE_TAR>']]
    return main_ticker_data

def prepadedata(main_ticker_data, train_seq, train_vol):
    # --- Разделяем данные на учебный и проверочный наборы
    input_train = []
    output_train = []
    input_test = []
    output_test = []

    # Получаем размерность массива
    data_row = main_ticker_data.shape[0]
    data_column = main_ticker_data.shape[1]
    # Переводим в формат np.array
    main_ticker_data = main_ticker_data.values

    # --- Нормализация
    main_ticker_data, data_mean, data_std = normax(main_ticker_data)

    print("main_ticker_data.shape: ", main_ticker_data.shape)
    # Тренировочные данные начинаются с 0 элемента массива
    train_start = 0
    # до train_vol*100% всего набора данных. На выходе получаем целое число от train_vol*data_row
    train_end = int(np.floor(train_vol * data_row))
    # Тестовые данные
    test_start = train_end + 1
    test_end = data_row
    data_train = main_ticker_data[np.arange(train_start, train_end), :]
    data_test = main_ticker_data[np.arange(test_start, test_end), :]

    # --- Подготовка учебного набора
    print("data_train_reverse.shape: ", data_train.shape)
    # Количество наборов в массиве с учетом смещения при комбинаторике
    row_train = (data_train.shape[0] // (train_seq + 1)) - (train_seq + 1)
    print("row_train: ", row_train)
    for z in range(0, train_seq):
        for i in range(0, row_train):
            x = np.array(data_train[i + z: i + z + train_seq])
            y = np.array(data_train[i + z + train_seq, 1:4])
            input_train.append(x)
            output_train.append(y)
    X_train = np.array(input_train)
    y_train = np.array(output_train)
    print("============")
    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", y_train.shape)
    print("============")
    #exit(0)

    # --- Подготовка тестового набора
    print("data_test.shape: ", data_test.shape)
    # Количество наборов в массиве с учетом смещения при комбинаторике
    row_test = (data_test.shape[0] // (train_seq + 1)) - (train_seq + 1)
    print("row_test: ", row_test)
    for z in range(0, train_seq):
        for i in range(0, row_test):
            xt = np.array(data_test[i + z: i + z + train_seq])
            yt = np.array(data_test[i + z + train_seq, 1:4])
            input_test.append(xt)
            output_test.append(yt)
    X_test = np.array(input_test)
    y_test = np.array(output_test)
    print("============")
    print("X_test.shape:", X_test.shape)
    print("y_test.shape:", y_test.shape)
    print("============")

    # --- Вывод
    return X_train, y_train, X_test, y_test, data_mean, data_std
    #return X_train, y_train


def normax(data):
    """
    :param data: Массив данных
    :return: Нормализованный массив data_return, среднее по столбцу data_mean, стандартное отклонение по столбцу data_std
    """
    data_return = data.astype(np.float64)  # Приведение типов
    data_mean = data.mean(axis=0)  # Вычисляем среднее по каждому столбцу
    data_std = data.std(axis=0)  # Определяем стандартное отклонение по каждому столбцу
    data_return -= data_mean  # Вычитаем среднее
    data_return /= data_std  # Делим на отклонение
    return data_return, data_mean, data_std


def normay(data, data_mean, data_std):
    """
    :param data: Массив данных
    :return: Нормализованный массив data_return, среднее по столбцу data_mean, стандартное отклонение по столбцу data_std
    """
    data_return = data.astype(np.float64)  # Приведение типов
    data_return -= data_mean  # Вычитаем среднее
    data_return /= data_std  # Делим на отклонение
    return data_return


def denorma(data, std, mean):
    """
    :param data: элемент массива/массив
    :param mean: среднее по столбцу
    :param std: стандартное отклонение по столбцу
    :return: нормализованные данные
    """
    data *= std     # Умножаем на стандартное отклонение для 0 столбца
    data += mean    # Прибавляем среднее для 0 столбца
    return data


def save(model, mse, mae, data_mean, data_std):
    """
    :param model:
    :return: Null
    """
    filedir = "models"
    now = datetime.datetime.now()
    ts = now.strftime("%d-%m-%Y_%H_%M")
    json_file = open(filedir + "/last_" + str(ts) + ".mse." + str(mse) + ".mae" + str(mae) + ".json", "w")
    json_file.write(model.to_json())
    json_file.close()
    model.save_weights(filedir + "/last_" + str(ts) + ".mse." + str(mse) + ".mae" + str(mae) + ".h5")
    paradata_file = open(filedir + "/last_" + str(ts) + ".mse." + str(mse) + ".mae" + str(mae) + ".txt", "w")
    paradata_file.write(str(data_mean) + "\n")
    paradata_file.write(str(data_std) + "\n")
    paradata_file.close()
