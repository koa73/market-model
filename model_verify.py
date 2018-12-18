import tensorflow as tf
import datama as d

model = "weights_1"

# Загрузка проверочных данных
data = d.DataManager("USDRUB", 5, 1)
X_test, y_test = data.get_test_data()

#print(data.de_norma(X_test))

# Загружаем сеть
json_file = open(data.get_current_dir()+"/models/"+model+".json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)                   # Создаем модель
loaded_model.load_weights(data.get_current_dir()+"/models/"+model+".h5")            # Загружаем веса
loaded_model.compile(loss='mse', optimizer='adam', metrics=['mae'])                 # Компилируем

mse, mae = loaded_model.evaluate(X_test, y_test)            #Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)

score = loaded_model.evaluate(X_test, y_test, verbose=0, batch_size=3)
print("Точность работы на тестовых данных : %.2f%%" % (score[1]*100))

predict = loaded_model.predict(X_test)    # Предсказания
data.predict_report(y_test, predict)

