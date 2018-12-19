import tensorflow as tf
import datama as d

model_name = "last_weights"

# Загрузка проверочных данных
data = d.DataManager("USDRUB", 5, 1)

# Загружаем сеть
json_file = open(data.get_current_dir()+"/models/"+model_name+".json", "r")
model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(model_json)                   # Создаем модель
model.load_weights(data.get_current_dir()+"/models/"+model_name+".hdf5")            # Загружаем веса
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                 # Компилируем

# Тестирование модели
X_test, y_test = data.get_test_data()
y_test_shaped = data.reshapy_y_by_coll(y_test, 1)


mse, mae = model.evaluate(X_test, y_test_shaped, verbose=0)            # Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)

predict = data.denorm_y_array(model.predict(X_test))    # Предсказания

print('--------------------------------------------------------')
print(predict)
print('--------------------------------------------------------')
print(data.denorm_y(y_test_shaped))
print('========================================================')
#data.predict_report(y_test_shaped, predict)

for i in range(len(y_test_shaped)):
    print(predict[i], y_test_shaped[i], "\t", predict[i][0]-y_test_shaped[i])

