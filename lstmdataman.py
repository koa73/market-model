import pandas as pd
import numpy as np

def loaddata(filename, separator):
    # Читаем файл
    main_ticker_data = pd.read_csv(filename, sep=separator)
    # Берем только нужные поля
    main_ticker_data = main_ticker_data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]
    return main_ticker_data

def prepadedata(main_ticker_data, train_seq, train_vol):
    # --- Разделяем данные на учебный и проверочный наборы

    # Получаем размерность массива
    data_row = main_ticker_data.shape[0]
    data_column = main_ticker_data.shape[1]
    # Переводим в формат np.array
    main_ticker_data = main_ticker_data.values
    # Тренировочные данные начинаются с 0 элемента массива
    train_start = 0
    # до 80% всего набора данных. На выходе получаем целое число от train_vol*data_row
    train_end = int(np.floor(train_vol * data_row))
    # Тестовые данные
    test_start = train_end + 1
    test_end = data_row
    data_train = main_ticker_data[np.arange(train_start, train_end), :]
    data_test = main_ticker_data[np.arange(test_start, test_end), :]
    # --- Нормализация учебного набора. Тестовый набор должен нормализоваться этими же параметрами mean и std
    data_train, data_mean, data_std = normax(data_train)
    data_test = normay(data_test, data_mean, data_std)
    # --- Делаем реверс наборов, для того, чтобы избежать удаления последних данных при целочисленном делении
    reverse_order = []
    for i in range(0, data_train.shape[0]):
        reverse_order = np.append(reverse_order, (data_train.shape[0] - 1) - i)
    data_train_reverse = data_train[reverse_order.astype(np.int64)]

    #for i in range(0, data_test.shape[0]):
    #    reverse_order = np.append(reverse_order, (data_test.shape[0] - 1) - i)
    #data_test_reverse = data_test[reverse_order.astype(np.int64)]

    # --- Подготовка учебного набора
    # Делим на последовательности "Кол-во измерений, кол-во шагов, кол-во элементов в измерении"
    # Находим количество строк, на сколько будет разбит новый массив
    print("data_train_reverse.shape[0]: ", data_train_reverse.shape[0])
    row_train = data_train_reverse.shape[0] // train_seq
    print("row_train: ", row_train)
    # Обрезаем массив для y
    data_train_reverse_y = data_train_reverse[:(row_train * train_seq)]
    data_train_reverse_y = data_train_reverse_y[:, 1:4]
    print("data_train_reverse_y.shape", data_train_reverse_y.shape)
    # Обрезаем массив для X. Смещаем на 1 строку, так как нам нужно иметь y_train для последнего набора данных и вычитаем последний набор для сохранения размерности
    data_train_reverse_x = data_train_reverse[1:(row_train * train_seq) - train_seq + 1]
    print("data_train_reverse_x.shape", data_train_reverse_x.shape)
    # Трансформируем массив
    X_train = data_train_reverse_x.reshape((row_train - 1), data_train_reverse_x.shape[1], train_seq)
    print("X_train.shape", X_train.shape)
    # y_train должен быть значением колонок <HIGH>;<LOW>;<CLOSE>. Последовательность 0, train_seq + 1 и т.д. с шагом train_seq
    print("row_train:", row_train)
    # Берем первую строку (предсказание для первого набора), колонки <HIGH>;<LOW>;<CLOSE>
    y_train = data_train_reverse[0, 1:4]
    #print("data_train_reverse_y: ", data_train_reverse_y)
    print("y_train:", y_train)
    for i in range(1, (row_train - 1)):
        #x = np.array(data_train_reverse_y.iloc[i: i + seq_len, 1])
        #print(data_train_reverse_y[i * (train_seq + 1), 1:3])
        y_train = np.append(y_train, data_train_reverse_y[i * (train_seq + 1), 1:4])
    print("y_train.shape: ", y_train.shape)
    # --- Подготовка тестового набора

    # --- Обратный реверс

    # ---
    #return X_train, y_train, X_test, y_test, data_mean, data_std
    return X_train, y_train


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