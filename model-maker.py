#!/usr/bin/env python3.5

import dataman_new as d
import tensorflow as tf
import keras.backend as K


data = d.DataManager("USDRUB", 5, 1)

X_train, y_train = data.get_edu_data()

model = tf.keras.Sequential()
print(X_train.shape)
model.add(tf.keras.layers.Dense(46, input_shape=(X_train.shape[1],), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(loss=['mse'], optimizer='adam', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=10, validation_split=0.01, verbose=2)      #Тренировка сети
#model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=2)

# Сохраняем сеть
data.save(model)


# Тестирование модели
X_test, y_test = data.get_test_data()
score = model.evaluate(X_test, y_test, verbose=0, batch_size=5)
print("Точность работы на тестовых данных : %.2f%%" % (score[1]*100))

predict = model.predict(X_test)    # Предсказания
data.predict_report(y_test, predict)

