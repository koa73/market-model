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
batch_size = 1
epochs = 3

X_train, y_train, X_test, y_test, data_mean, data_std = lstmdataman.prepadedata(main_ticker_data, train_seq, train_vol)

print("data_mean: ", data_mean)
print("data_std: ", data_std)
#exit(0)

"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(512, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu'))
model.add(tf.keras.layers.Dense(3))
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(512, activation='relu', dropout=0.3, return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu'))
model.add(tf.keras.layers.Dense(3))


model.compile(loss='mse', optimizer='adam', metrics=['mae'])

print("\n====== Train ======\n")
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)      #Тренировка сети

print("\n====== Test ======\n")
mse, mae = model.evaluate(X_test, y_test) #Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)

pred = model.predict(X_test)    # Предсказания

#print(pred)

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

# Сохраняем сеть
#lstmdataman.save(model, mse, mae, data_mean, data_std)



