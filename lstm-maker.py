import tensorflow as tf
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import lstmdataman

filename = 'data/USDRUB.csv'
separator = ';'
main_ticker_data = lstmdataman.loaddata(filename, separator)

train_vol = 0.8         # Сколько берем от объема для обучения
train_seq = 3           # Непрерывная последовательность, для которой будем искать предсказание: Х дня -> 1 ответ

X_train, y_train, X_test, y_test, data_mean, data_std = lstmdataman.prepadedata(main_ticker_data, train_seq, train_vol)
#X_train, y_train = lstmdataman.prepadedata(main_ticker_data, train_seq, train_vol)

print("data_mean: ", data_mean)
print("data_std: ", data_std)
#exit(0)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(512, activation='relu', dropout=0.1, return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', dropout=0.1, return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu', dropout=0.1))
model.add(tf.keras.layers.Dense(3))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

print("\n====== Train ======\n")
model.fit(X_train, y_train, epochs=1, batch_size=10)      #Тренировка сети

print("\n====== Test ======\n")
mse, mae = model.evaluate(X_train, y_train) #Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)
pred = model.predict(X_test)    # Предсказания

#print(pred)

#Денормализация
last_pred = pred[len(y_test) - 1]    #Последнее предсказание
last_pred *= data_std[1:4]          #Умножаем на стандартное отклонение для 0 столбца
last_pred += data_mean[1:4]         #Прибавляем среднее для 0 столбца
last_y = y_test[len(y_test) - 1]
last_y *= data_std[1:4]
last_y += data_mean[1:4]

first_pred = pred[0]                 #Первое предсказание
first_pred *= data_std[1:4]          #Умножаем на стандартное отклонение для 0 столбца
first_pred += data_mean[1:4]        #Прибавляем среднее для 0 столбца
first_y = y_test[0]
first_y *= data_std[1:4]
first_y += data_mean[1:4]

middle_pred = pred[len(y_test) // 2]  #Среднее предсказание
middle_pred *= data_std[1:4]           #Умножаем на стандартное отклонение для 0 столбца
middle_pred += data_mean[1:4]          #Прибавляем среднее для 0 столбца
middle_y = y_test[len(y_test) // 2]
middle_y *= data_std[1:4]
middle_y += data_mean[1:4]

# Вывод результатов прогона по тестовому массиву - первый, середина и последний
print(first_pred, first_y, (first_pred - first_y))
print(middle_pred, middle_y, (middle_pred - middle_y))
print(last_pred, last_y, (last_pred - last_y))

# Сохраняем сеть
lstmdataman.save(model, mse, mae, data_mean, data_std)



