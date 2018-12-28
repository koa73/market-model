import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import os
import datetime


def loadfile(ticker, market_identifier, start_date, end_date, separator=","):
    print("Load data from MOEX, ticker:", ticker)
    raw_data = pdr.DataReader(ticker, 'moex', start_date, end_date)
    # Выбираем нужный индекс (Идентификатор режима торгов)
    select_indices = list(np.where(raw_data['BOARDID'] == market_identifier)[0])
    raw_data = raw_data.iloc[select_indices]
    raw_data = raw_data[['OPEN', 'LOW', 'HIGH', 'CLOSE', 'VALUE', 'VOLUME']]
    # Сохраняем файл
    raw_data.to_csv(ticker + '.csv')
    print("Load data complete\n")


def loaddata(ticker, separator):
    # Читаем файл
    main_ticker_data = pd.read_csv(ticker + '.csv', sep=separator)
    # Берем только нужные поля
    #main_ticker_data = main_ticker_data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
    #main_ticker_data = main_ticker_data[['<OPEN_TAR>', '<HIGH_TAR>', '<LOW_TAR>', '<CLOSE_TAR>', '<VOL_TAR>', '<OPEN_DEP1>', '<HIGH_DEP1>', '<LOW_DEP1>', '<CLOSE_DEP1>']]
    main_ticker_data = main_ticker_data[['OPEN', 'LOW', 'HIGH', 'CLOSE', 'VALUE', 'VOLUME']]
    # Переводим из формата pandas.DataFrame в np.array
    return main_ticker_data.values


def prepadedata(main_ticker_data, train_seq, train_vol):
    # --- Разделяем данные на учебный и проверочный наборы
    input_train = []
    output_train = []
    input_test = []
    output_test = []

    # --- Нормализация
    main_ticker_data = norma(main_ticker_data)

    # Получаем размерность массива
    data_row = main_ticker_data.shape[0]

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
    #
    print("data_train.shape: ", data_train.shape)
    print("data_test.shape: ", data_test.shape)

    # --- Подготовка учебного набора
    for i in range(0, train_seq):
        z = 1
        while z < (data_train.shape[0] - (2 * train_seq)):
            x = np.array(data_train[i + z: i + z + train_seq])
            # y = np.array(data_train[i + z + train_seq, 1:4])  #Close, High, Low
            y = np.array(data_train[i + z + train_seq, 3])  # Close
            z = z + train_seq
            input_train.append(x)
            output_train.append(y)
    X_train = np.array(input_train)
    y_train = np.array(output_train)
    print("============")
    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", y_train.shape)
    print("============")
    # exit(0)

    # --- Подготовка тестового набора
    #for i in range(0, train_seq):
    z = 1
    while z < (data_test.shape[0] - (2 * train_seq)):
        xt = np.array(data_test[i + z: i + z + train_seq])
        # y = np.array(data_test[i + z + train_seq, 1:4])  #Close, High, Low
        yt = np.array(data_test[i + z + train_seq, 3])  # Close
        z = z + train_seq
        input_test.append(xt)
        output_test.append(yt)
    X_test = np.array(input_test)
    y_test = np.array(output_test)
    print("============")
    print("X_test.shape:", X_test.shape)
    print("y_test.shape:", y_test.shape)
    print("============")

    # --- Вывод
    return X_train, y_train, X_test, y_test


def norma(main_ticker_data):
    """
    :param data: Массив данных
    :return: Нормализованный массив data_return, среднее по столбцу data_mean, стандартное отклонение по столбцу data_std
    """
    main_ticker_data = main_ticker_data.astype(np.float64)  # Приведение типов
    result_arr = []
    tmp_arr = np.empty((0, 6), np.float64)

    for i in range(1, main_ticker_data.shape[0]):
        # OPEN,LOW,HIGH,CLOSE,VALUE,VOLUME
        #  [0] [1] [2]   [3]   [4]   [5]
        tmp_arr = [[main_ticker_data[i][0] / main_ticker_data[i - 1][0]],
                   [main_ticker_data[i][1] / main_ticker_data[i - 1][1]],
                   [main_ticker_data[i][2] / main_ticker_data[i - 1][2]],
                   [main_ticker_data[i][3] / main_ticker_data[i - 1][3]],
                   [main_ticker_data[i][4] / main_ticker_data[i - 1][4]],
                   [main_ticker_data[i][5] / main_ticker_data[i - 1][5]]]
        # print(tmp_arr)
        result_arr.append(tmp_arr)

    result_arr = np.array(result_arr)
    result_arr = np.reshape(result_arr, (result_arr.shape[0], result_arr.shape[1]))
    print("result_arr.shape: ", result_arr.shape)
    return result_arr


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
    filedir = "../models"
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
    print("/last_" + str(ts) + ".mse." + str(mse) + ".mae" + str(mae))

