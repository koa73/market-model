import tensorflow as tf
import numpy as np
import lstm-dataman



#data = d.DataManager("USDRUB.csv", 5, 1)

#X_train, y_train = data.get_edu_data()

#print(data.get_variations_num())


X_train = np.array([
    [66.1340000, 66.4600000, 65.7151000, 66.0181000, 62380],
    [66.0570000, 66.1553000, 65.6437000, 66.0930000, 330417],

    [66.1226000, 66.3300000, 65.7775000, 66.3100000, 364681],
    [66.3261000, 66.9589000, 66.0907000, 66.8795000, 464003],

    [66.8748000, 68.1942000, 66.7153000, 67.9080000, 503393],
    [67.9025000, 67.9872000, 67.0565000, 67.8863000, 383317],
#    [67.8914000, 68.1536000, 67.4866000, 68.1015000, 521420]
    ])

#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) # 8x5x1 "Кол-во измерений, кол-во шагов, кол-во элементов в измерении"
X_train = X_train.reshape(3, X_train.shape[1], 2) # 8x5x1 "Кол-во измерений, кол-во шагов, кол-во элементов в измерении"

print(X_train.shape, X_train.shape[0], X_train.shape[1], X_train.shape[2])

print(X_train)
y_train = np.array([
    [66.3300000, 65.7775000, 66.3100000],

    [68.1942000, 66.7153000, 67.9080000],

    [68.1536000, 67.4866000, 68.1015000],
#    [68.2667000, 66.9588000, 67.0521000]
])
#y_train = y_train.reshape(1, y_train.shape[0], y_train.shape[1])
print(y_train.shape)
print(y_train)
#exit(0)

"""
На вход LSTM сети подается трехмерный массив размерности "Кол-во измерений, кол-во шагов, кол-во элементов в измерении"
На выходе ожидается массив такой же размерности
"""
#exit(0)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(256, activation='relu', dropout=0.1, return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu', dropout=0.1, return_sequences=True))
model.add(tf.keras.layers.LSTM(64, activation='relu', dropout=0.1))
model.add(tf.keras.layers.Dense(3))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(X_train, y_train, epochs=2, batch_size=1)      #Тренировка сети

print("\n====== Test ======\n")
mse, mae = model.evaluate(X_train, y_train) #Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mse)

# Сохраняем сеть
#prefix = "mixa"
#data.save(model, prefix)



