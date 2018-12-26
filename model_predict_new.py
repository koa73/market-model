import tensorflow as tf
import datama_new as d

model_name = "weights_100"

# Загрузка проверочных данных
data = d.DataManager("USDRUB_TOM_2", 5, 1)

# Загружаем сеть
json_file = open(data.get_current_dir()+"/models/"+model_name+".json", "r")
model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(model_json)                   # Создаем модель
model.load_weights(data.get_current_dir()+"/models/"+model_name+".h5")            # Загружаем веса
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                 # Компилируем

# Тестирование модели
X_test = data.get_predict_data()

predict = model.predict(X_test)
data.predict_report(predict)                                            # Предсказания


