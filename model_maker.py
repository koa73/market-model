#!/usr/bin/env python3

import tensorflow as tf
import sys
import modelMaker as d
import numpy as np

data = d.ModelMaker()

print("Start model making ....")

if (len(sys.argv) < 3):
    print("Argument not found ")
    exit(0)

X_UP, y_UP = data.get_edu_data('edu','UP_'+sys.argv[1], '2D')
X_DOWN, y_DOWN = data.get_edu_data('edu','UP_'+sys.argv[1], '2D')
X_NONE, y_NONE = data.get_edu_data('edu','UP_'+sys.argv[1], '2D')

class_weight = {0: 1., 1: 1., 2: 1.}
class_weight[1] = float(sys.argv[2])

X_train = np.concatenate((X_DOWN,X_UP), axis=0)
y_train = np.concatenate((y_DOWN,y_UP), axis=0)
X_train = np.concatenate((X_train,X_NONE), axis=0)
y_train = np.concatenate((y_train,y_NONE), axis=0)

input_layer_1 = tf.keras.layers.Input(shape=(24,))
norma_layer = tf.keras.layers.LayerNormalization(axis=1)(input_layer_1)
hidden_d2_dense = tf.keras.layers.Dense(12, activation='tanh')(norma_layer)
hidden_d3_dense = tf.keras.layers.Dense(34, activation='tanh')(hidden_d2_dense)
hidden_d4_dense = tf.keras.layers.Dense(34, activation='tanh')(hidden_d3_dense)
hidden_d5_dense = tf.keras.layers.Dense(6, activation='tanh')(hidden_d4_dense)
output = tf.keras.layers.Dense(3, activation='softmax')(hidden_d5_dense)

#
model = tf.keras.models.Model(inputs=[input_layer_1], outputs=[output])
#
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
print(model.summary())
data.save_conf(model, sys.argv[1])                                                  # Запись конфигурации скти для прерывания расчета

dirPath = data.get_file_dir()+ "/data/model_test/"

# Сохранение модели с лучшими параметрами
checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor='accuracy',
                                                  filepath = dirPath + "weights_"+sys.argv[1]+".h5",
                                                  verbose=1, save_best_only=True)
# Уменьшение коэфф. обучения при отсутствии изменения ошибки в течении learn_count эпох
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=5, min_lr=0.000001,
                                                 verbose=1)
# Остановка при переобучении. patience - сколько эпох мы ждем прежде чем прерваться.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='auto')

# Тренировка сети
model.fit(X_train, y_train, class_weight=class_weight, validation_split=0.05, epochs=100,
batch_size=10, verbose=1, shuffle=True,callbacks=[checkpointer, reduce_lr, early_stopping])
