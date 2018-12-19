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

train_vol = 0.9         # Сколько берем от объема для обучения
train_seq = 4           # Непрерывная последовательность, для которой будем искать предсказание: Х дня -> 1 ответ
batch_size = 10
epochs = 4

X_train, y_train, X_test, y_test, data_mean, data_std = lstmdataman.prepadedata(main_ticker_data, train_seq, train_vol)

print("data_mean: ", data_mean)
print("data_std: ", data_std)
#exit(0)

"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu'))
model.add(tf.keras.layers.Dense(3))
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, bias_initializer='zeros'))
model.add(tf.keras.layers.LSTM(512, activation='relu', dropout=0.2, return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu', dropout=0.2))
model.add(tf.keras.layers.Dense(3))


model.compile(loss='mse', optimizer='adam', metrics=['mae'])

print("\n====== Train ======\n")
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)      #Тренировка сети

print("\n====== Test ======\n")
mse, mae = model.evaluate(X_test, y_test) #Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)

pred = model.predict(X_test)    # Предсказания

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

# Сохраняем сеть
lstmdataman.save(model, mse, mae, data_mean, data_std)

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




