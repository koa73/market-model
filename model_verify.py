import tensorflow as tf
import datama as d
import matplotlib.pyplot as plt

model_name = "weights"

# Загрузка проверочных данных
data = d.DataManager("USDRUB", 4, 1)

# Загружаем сеть
json_file = open(data.get_current_dir()+"/models/"+model_name+".json", "r")
model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(model_json)                   # Создаем модель
model.load_weights(data.get_current_dir()+"/models/"+model_name+".h5")            # Загружаем веса
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                 # Компилируем

# Тестирование модели
X_test, y_test = data.get_test_data()
y_test_shaped = data.reshapy_y_by_coll(y_test, 1)


mse, mae = model.evaluate(X_test, y_test_shaped, verbose=0)            # Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)

predict = model.predict(X_test)    # Предсказания

print('--------------------------------------------------------')
print(predict)
print('--------------------------------------------------------')
print(y_test_shaped)
print('========================================================')
#data.predict_report(y_test_shaped, predict)

for i in range(len(y_test_shaped)):
    print(predict[i], y_test_shaped[i], "\t", predict[i][0]-y_test_shaped[i])

try:
    X_p, y_p = data.get_graph_data()
    predict = model.predict(X_p)
    y_test_shaped = data.reshapy_y_by_coll(y_p, 1)
    # Отображение данных
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.grid()
    ax1.set(xlabel='time (Day)', ylabel='High (USDRUB)', title='Cost')
    ax1.legend()

    line1, = ax1.plot(y_test_shaped,
                      label="High real")  # Отображаем на графике тестовые данные по колонке 0 по y_test (HIGH)
    line2, = ax1.plot(predict, label="High predict")  # Предсказания по колонке 0 оранжевым цветом

    ax1.legend()
    plt.show()
    plt.waitforbuttonpress()

except Exception:
    None

