import tensorflow as tf
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import lstmdataman
import matplotlib.pyplot as plt

"""
Утилита для загрузки и подготовки данных для индекса
"""

market_identifier = 'TQBR'
start_date = '2010-01-01'
end_date = '2018-12-10'
separator = ','

#lstmdataman.loadfile('ALRS', market_identifier, start_date, end_date)
#lstmdataman.loadfile('CHMF', market_identifier, start_date, end_date)
#lstmdataman.loadfile('FIVE', market_identifier, start_date, end_date)
#lstmdataman.loadfile('GAZP', market_identifier, start_date, end_date)
#lstmdataman.loadfile('GMKN', market_identifier, start_date, end_date)
#lstmdataman.loadfile('LKOH', market_identifier, start_date, end_date)
#lstmdataman.loadfile('MGNT', market_identifier, start_date, end_date)
#lstmdataman.loadfile('MTSS', market_identifier, start_date, end_date)
#lstmdataman.loadfile('NVTK', market_identifier, start_date, end_date)
#lstmdataman.loadfile('ROSN', market_identifier, start_date, end_date)
#lstmdataman.loadfile('SBER', market_identifier, start_date, end_date)
#lstmdataman.loadfile('SNGS', market_identifier, start_date, end_date)
#lstmdataman.loadfile('TATN', market_identifier, start_date, end_date)
#lstmdataman.loadfile('VTBR', market_identifier, start_date, end_date)
#lstmdataman.loadfile('YNDX', market_identifier, start_date, end_date)

main_ticker_data = lstmdataman.loadbatchdata('data/ALRS', separator)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/CHMF', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/FIVE', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/GAZP', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/GMKN', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/LKOH', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/MGNT', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/MTSS', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/NVTK', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/ROSN', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/SBER', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/SNGS', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/TATN', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/VTBR', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)
main_ticker_data = np.append(main_ticker_data, lstmdataman.loaddata('data/YNDX', separator), axis=0)
print("main_ticker_data.shape: ", main_ticker_data.shape)

result_arr = []
prepare_arr = []
tmp_arr = np.empty((0, 6), float)

for i in range(0, main_ticker_data.shape[0] - 1):
    # OPEN,LOW,HIGH,CLOSE,VALUE,VOLUME
    #  [0] [1] [2]   [3]   [4]   [5]
    tmp_arr = [[main_ticker_data[i+1][0] / main_ticker_data[i][0]], [main_ticker_data[i+1][1] / main_ticker_data[i][1]],
               [main_ticker_data[i+1][2] / main_ticker_data[i][2]], [main_ticker_data[i+1][3] / main_ticker_data[i][3]],
               [main_ticker_data[i+1][4] / main_ticker_data[i][4]], [main_ticker_data[i+1][5] / main_ticker_data[i][5]]]
    #print(tmp_arr)
    result_arr.append(tmp_arr)

result_arr = np.array(result_arr)
result_arr = np.reshape(result_arr, (result_arr.shape[0], result_arr.shape[1]))
print("result_arr.shape: ", result_arr.shape)
prepare_arr = result_arr


train_vol = 0.9         # Сколько берем от объема для обучения
train_seq = 1           # Непрерывная последовательность, для которой будем искать предсказание: Х дня -> 1 ответ
batch_size = 10
epochs = 1

X_train, y_train, X_test, y_test, data_mean, data_std = lstmdataman.prepadedata(prepare_arr, train_seq, train_vol)

print("data_mean: ", data_mean)
print("data_std: ", data_std)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu'))
#model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.Dense(1))


#model.compile(loss='mse', optimizer='RMSprop', metrics=['mae'])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

print("\n====== Train ======\n")
#model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.01, verbose=1)      #Тренировка сети
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)      #Тренировка сети

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
    #last_pred *= data_std[3]          #Умножаем на стандартное отклонение для 0 столбца
    #last_pred += data_mean[3]         #Прибавляем среднее для 0 столбца
    #last_y *= data_std[3]
    #last_y += data_mean[3]
    # Вывод результатов прогона по тестовому массиву - первый, середина и последний
    print(last_pred, last_y, (last_pred - last_y))
    p_input_test.append(last_pred)
    y_output_test.append(last_y)


pred_test_plot = np.array(p_input_test)
y_test_plot = np.array(y_output_test)

# Сохраняем сеть
#lstmdataman.save(model, mse, mae, data_mean, data_std)

# Отображение данных
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid()
ax1.set(xlabel='time (Day)', ylabel='Close (RUB)', title='Cost')
ax1.legend()

line1, = ax1.plot(y_test_plot[np.arange(0, len(y_test) - 1)], label="Tst_Clo")  #Отображаем на графике тестовые данные по колонке 0 по y_test (HIGH)
line2, = ax1.plot(pred_test_plot[np.arange(0, len(y_test) - 1)], label="Prd_Clo")  # Предсказания по колонке 0 оранжевым цветом

ax1.legend()

plt.show()
plt.waitforbuttonpress()

