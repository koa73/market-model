#!/usr/bin/env python3

import tensorflow as tf
import sys
import dataMiner as D
import ConcatLayer_3 as c
import Antirictifier as A

data = D.DataMiner(3)

print("Start model making ....")

if (len(sys.argv) < 2):
    print("Argument not found ")
    #exit(0)

X_train = data.get_edu('edu_X_', 'b22_200')
X_train = X_train.reshape(X_train.shape[0],-1)
y_train =  data.get_edu('edu_y_', 'b22_200')

print("X_edu : " + str(X_train.shape))
print("y_edu : " + str(y_train.shape))

print(X_train.shape)

input_layer_1 = tf.keras.layers.Input(shape=(24,))
norma_layer = tf.keras.layers.LayerNormalization(axis=1)(input_layer_1)
hidden_d2_dense = tf.keras.layers.Dense(12, activation='tanh')(norma_layer)
hidden_d5_dense = tf.keras.layers.Dense(9, activation='tanh')(hidden_d2_dense)
hidden_d6_dense = c.ConcatLayer(10)(hidden_d5_dense)
hidden_d7_dense = tf.keras.layers.Dense(24, activation='tanh')(hidden_d6_dense)
hidden_d8_dense = tf.keras.layers.Dense(12, activation='tanh')(hidden_d7_dense)
output = tf.keras.layers.Dense(3, activation='softmax')(hidden_d8_dense)

#
model = tf.keras.models.Model(inputs=[input_layer_1], outputs=[output])
#
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
print(model.summary())
data.save_conf(model, 'b22_200')                                                  # Запись конфигурации скти для прерывания расчета

dirPath = data.get_current_dir()+ "/data/model_test/"

# Сохранение модели с лучшими параметрами
checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor='accuracy',
                                                  filepath = dirPath + "weights_"+'b22_200'+".h5",
                                                  verbose=1, save_best_only=True)
# Уменьшение коэфф. обучения при отсутствии изменения ошибки в течении learn_count эпох
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=5, min_lr=0.000001,
                                                 verbose=1)
# Остановка при переобучении. patience - сколько эпох мы ждем прежде чем прерваться.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='auto')

# Тренировка сети
model.fit(X_train, y_train, validation_split=0.05, epochs=100,
batch_size=10, verbose=1, shuffle=True,callbacks=[checkpointer, reduce_lr, early_stopping])
