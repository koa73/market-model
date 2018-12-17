import tensorflow as tf
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import lstmdataman
import matplotlib.pyplot as plt

filename = 'data/USDRUB.csv'
#filename = "data/COMMON_SBER-USDCB_100101_181207.csv"
separator = ';'
#separator = ','
main_ticker_data = lstmdataman.loaddata(filename, separator)

filedir = "models/"
model = "last_17-12-2018_10_30.mse.0.053883990893761315.mae0.19663288791974384"

train_vol = 0.9         # Сколько берем от объема для обучения
train_seq = 4           # Непрерывная последовательность, для которой будем искать предсказание: Х дня -> 1 ответ
batch_size = 5
epochs = 3

X_train, y_train, X_test, y_test, data_mean, data_std = lstmdataman.prepadedata(main_ticker_data, train_seq, train_vol)

print("data_mean: ", data_mean)
print("data_std: ", data_std)

# Загружаем сеть
json_file = open(filedir + model + ".json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)                   # Создаем модель
loaded_model.load_weights(filedir + model + ".h5")            # Загружаем веса
loaded_model.compile(loss='mse', optimizer='adam', metrics=['mae'])                 # Компилируем

mse, mae = loaded_model.evaluate(X_test, y_test)            #Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)

pred = loaded_model.predict(X_test)    # Предсказания

p_input_test = []
y_output_test = []

#Денормализация
for i in range(0, len(y_test) - 1):
    last_pred = pred[i]
    last_y = y_test[i]
    last_pred *= data_std[1:4]          #Умножаем на стандартное отклонение для 0 столбца
    last_pred += data_mean[1:4]         #Прибавляем среднее для 0 столбца
    last_y *= data_std[1:4]
    last_y += data_mean[1:4]
    # Вывод результатов прогона по тестовому массиву - первый, середина и последний
    print(last_pred, last_y, (last_pred - last_y))
    p_input_test.append(last_pred)
    y_output_test.append(last_y)

pred_test_plot = np.array(p_input_test)
y_test_plot = np.array(y_output_test)

# Отображение данных
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid()
ax1.set(xlabel='time (Day)', ylabel='Close (RUB)', title='Cost')
ax1.legend()

line1, = ax1.plot(y_test_plot[np.arange(0, len(y_test) - 1), 0], label="Tst_HI")  #Отображаем на графике тестовые данные по колонке 0 по y_test (HIGH)
line2, = ax1.plot(pred_test_plot[np.arange(0, len(y_test) - 1), 0], label="Prd_HI")  # Предсказания по колонке 0 оранжевым цветом

line3, = ax1.plot(y_test_plot[np.arange(0, len(y_test) - 1), 1], label="Tst_Lo")  #Отображаем на графике тестовые данные по колонке 1 по y_test (LOW)
line4, = ax1.plot(pred_test_plot[np.arange(0, len(y_test) - 1), 1], label="Prd_Lo")  # Предсказания по колонке 1 красным цветом

line5, = ax1.plot(y_test_plot[np.arange(0, len(y_test) - 1), 2], label="Tst_Clo")  #Отображаем на графике тестовые данные по колонке 1 по y_test (Close)
line6, = ax1.plot(pred_test_plot[np.arange(0, len(y_test) - 1), 2], label="Prd_Clo")  # Предсказания по колонке 1

ax1.legend()


plt.show()
plt.waitforbuttonpress()

