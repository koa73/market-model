#!/usr/bin/env python3.5

import dataman_new as d
import tensorflow as tf


data = d.DataManager("USDRUB", 5, 2)

X_train, y_train = data.get_edu_data()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(135, input_shape=(X_train.shape[1],), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(123, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(116, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.05, verbose=2)      #Тренировка сети
#model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=2)

# Сохраняем сеть
data.save(model)

#Проверка обучкемя
X_validate, y_validate = data.get_validation_data()
score = model.evaluate(X_validate, y_validate, verbose=0)
print("Точность работы на проверочных данных : %.2f%%" % (score[1]*100))

#Тестирование модели
X_test, y_test = data.get_test_data()
score = model.evaluate(X_test, y_test, verbose=0)
print("Точность работы на тестовых данных : %.2f%%" % (score[1]*100))

