#!/usr/bin/env python3

import tensorflow as tf
import sys
import dataMiner as D

data = D.DataMiner(3)

print("Start model making ....")

if (len(sys.argv) < 2):
    print("Argument not found ")
    exit(0)

X_train = data.get_edu('edu_X_', sys.argv[1])
X_train = X_train.reshape(X_train.shape[0],-1)
y_train =  data.get_edu('edu_y_', sys.argv[1])

print("X_edu : " + str(X_train.shape))
print("y_edu : " + str(y_train.shape))

print(X_train.shape)

input_layer_1 = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
norma_layer = tf.keras.layers.LayerNormalization(axis=1)(input_layer_1)
hidden_d2_dense = tf.keras.layers.Dense(48, activation='tanh')(norma_layer)
hidden_d3_dense = tf.keras.layers.Dense(96, activation='tanh')(hidden_d2_dense)
hidden_d4_dense = tf.keras.layers.Dense(24, activation='tanh')(hidden_d3_dense)
hidden_d5_dense = tf.keras.layers.Dense(6, activation='tanh')(hidden_d4_dense)
output = tf.keras.layers.Dense(3, activation='softmax')(hidden_d5_dense)

#
model = tf.keras.models.Model(inputs=[input_layer_1], outputs=[output])
#
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
print(model.summary())

data.save_conf(model, sys.argv[1])                                                  # Запись конфигурации скти для прерывания расчета
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])


dirPath = data.get_current_dir()+ "/data/model_test/"
# Сохранение модели с лучшими параметрами
checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor='accuracy',
                                                  filepath = dirPath + "weights_"+sys.argv[1]+".h5",
                                                  verbose=1, save_best_only=True)
# Уменьшение коэфф. обучения при отсутствии изменения ошибки в течении learn_count эпох
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=5, min_lr=0.000001,
                                                 verbose=1)
# Остановка при переобучении. patience - сколько эпох мы ждем прежде чем прерваться.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='auto')

# Тренировка сети
model.fit(X_train, y_train, validation_split=0.05, epochs=100,
          batch_size=10, verbose=1, shuffle=True,
          callbacks=[checkpointer, reduce_lr, early_stopping])
