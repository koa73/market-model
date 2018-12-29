import tensorflow as tf
import numpy as np
import lstmdatamanru
import matplotlib.pyplot as plt


path = 'data/'
ticker = 'SBER'
market_identifier = 'TQBR'
start_date = '2010-01-01'
end_date = '2018-12-10'
separator = ','


#lstmdataman.loadfile(path, ticker, market_identifier, start_date, end_date)
main_ticker_data = lstmdatamanru.loaddata(path, ticker, separator)

train_vol = 0.9         # Сколько берем от объема для обучения
train_seq = 1           # Непрерывная последовательность, для которой будем искать предсказание: Х дня -> 1 ответ
batch_size = 5
epochs = 5

X_train, y_train, X_test, y_test, data_mean, data_std = lstmdatamanru.prepadedata(main_ticker_data, train_seq, train_vol)

print("data_mean: ", data_mean)
print("data_std: ", data_std)


"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu'))
model.add(tf.keras.layers.Dense(3))
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(512, activation='relu', dropout=0.1, return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu'))
model.add(tf.keras.layers.Dense(1))


#model.compile(loss='mse', optimizer='RMSprop', metrics=['mae'])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

print("\n====== Train ======\n")
checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor='loss', filepath="models\weights.h5", verbose=1, save_best_only=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10, min_lr=0.000001, verbose=1)
#model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.01, verbose=1)      #Тренировка сети
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[checkpointer, reduce_lr])      #Тренировка сети

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
    last_pred *= data_std[3]          #Умножаем на стандартное отклонение для 0 столбца
    last_pred += data_mean[3]         #Прибавляем среднее для 0 столбца
    last_y *= data_std[3]
    last_y += data_mean[3]
    # Вывод результатов прогона по тестовому массиву - первый, середина и последний
    print(last_pred, last_y, (last_pred - last_y))
    p_input_test.append(last_pred)
    y_output_test.append(last_y)


pred_test_plot = np.array(p_input_test)
y_test_plot = np.array(y_output_test)

# Сохраняем сеть
lstmdatamanru.save(model, mse, mae, data_mean, data_std)

# Отображение данных
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid()
ax1.set(xlabel='time (Day)', ylabel='Close (RUB)', title='Cost')
ax1.legend()

#line1, = ax1.plot(y_test_plot[np.arange(0, len(y_test) - 1), 0], label="Tst_HI")  #Отображаем на графике тестовые данные по колонке 0 по y_test (HIGH)
line1, = ax1.plot(y_test_plot[np.arange(0, len(y_test) - 1)], label="Tst_Clo")  #Отображаем на графике тестовые данные по колонке 0 по y_test (HIGH)
#line2, = ax1.plot(pred_test_plot[np.arange(0, len(y_test) - 1), 0], label="Prd_HI")  # Предсказания по колонке 0 оранжевым цветом
line2, = ax1.plot(pred_test_plot[np.arange(0, len(y_test) - 1)], label="Prd_Clo")  # Предсказания по колонке 0 оранжевым цветом

#line3, = ax1.plot(y_test_plot[np.arange(0, len(y_test) - 1), 1], label="Tst_Lo")  #Отображаем на графике тестовые данные по колонке 1 по y_test (LOW)
#line4, = ax1.plot(pred_test_plot[np.arange(0, len(y_test) - 1), 1], label="Prd_Lo")  # Предсказания по колонке 1 красным цветом

#line5, = ax1.plot(y_test_plot[np.arange(0, len(y_test) - 1), 2], label="Tst_Clo")  #Отображаем на графике тестовые данные по колонке 1 по y_test (Close)
#line6, = ax1.plot(pred_test_plot[np.arange(0, len(y_test) - 1), 2], label="Prd_Clo")  # Предсказания по колонке 1

ax1.legend()


plt.show()
plt.waitforbuttonpress()





