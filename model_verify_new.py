import tensorflow as tf
import datama_new as d
import matplotlib.pyplot as plt

model_name = "weights"

# Загрузка проверочных данных
data = d.DataManager("USDRUB_TOM_2", 4, 1)

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

#predict = data.denorm_y_array(model.predict(X_test))    # Предсказания
predict = model.predict(X_test)    # Предсказания

X_test = data.get_test_denorm_data()

print('========================================================')

for i in range(len(y_test_shaped)):
    print(X_test[i][11:12], predict[i] / 10 * X_test[i][11:12], y_test_shaped[i] / 10 * X_test[i][11:12],
          predict[i] / 10 * X_test[i][11:12] - y_test_shaped[i] / 10 * X_test[i][11:12])
    #print(X_test[i][16:17], predict[i]/10*X_test[i][16:17], y_test_shaped[i]/10*X_test[i][16:17], predict[i]/10*X_test[i][16:17]-y_test_shaped[i]/10*X_test[i][16:17])
    #print(predict[i], y_test_shaped[i], "\t", predict[i][0]-y_test_shaped[i])


