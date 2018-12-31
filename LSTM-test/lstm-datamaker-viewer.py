import tensorflow as tf
import numpy as np
import lstmdatamanen
import matplotlib.pyplot as plt

path = 'data/test_'
filedir = 'models'
modelname = 'last_31-12-2018_11_38.mse.0.34912893385970944.mae0.424999559278468'
ticker = 'AAPL'
start_date = '2018-12-01'
end_date = '2018-12-20'
separator = ','

train_seq = 1           # Непрерывная последовательность, для которой будем искать предсказание: Х дня -> 1 ответ
batch_size = 1          # Нужно подбирать размер батча в зависимости от количества загружаемых данных
epochs = 10

lstmdatamanen.loadfileen(path, ticker, start_date, end_date)
test_ticker_data = lstmdatamanen.loaddataen(path, ticker, separator)

X_test, y_test, data_mean, data_std = lstmdatamanen.preparetestdata(test_ticker_data, train_seq)

print("data_mean: ", data_mean)
print("data_std: ", data_std)

print("\n====== Load Model ======\n")

# Загружаем сеть
json_file = open(filedir + "/" + modelname + ".json", "r")
model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(model_json)                   # Создаем модель
model.load_weights(filedir + "/" + modelname + ".h5")            # Загружаем веса
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                 # Компилируем


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

ax1.legend()
plt.show()
plt.waitforbuttonpress()

