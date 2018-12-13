import tensorflow as tf
import dataman as d

model = "last_12-12-2018_17_51"

# Загрузка проверочных данных
data = d.DataManager("USDRUB.csv", 5, 1)
X_test, y_test = data.get_edu_data()

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

print("=====================================")
predict = data.de_norma(loaded_model.predict(X_test))    # Предсказания
y_test_denorm = data.de_norma(y_test)

for i in range(len(y_test)):
    print(predict[i], y_test_denorm[i], "\t", [y_test_denorm[i][0]-predict[i][0], predict[i][1]-y_test_denorm[i][1]])
