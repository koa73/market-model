#!/usr/bin/env python3

import tensorflow as tf
import sys
import modelMaker as d
import numpy as np


data = d.ModelMaker()

print("Start model making ....")

if (len(sys.argv) < 2):
    print("Argument not found ")
    exit(0)

X_up, y_up = data.get_edu_data('edu','UP_'+sys.argv[1], '2D')
X_down, y_down = data.get_edu_data('edu','DOWN_'+sys.argv[1], '2D')
X_none, y_none = data.get_edu_data('edu','NONE_'+sys.argv[1], '2D')

class_weight = {0: 1., 1: 1., 2: 1.}

X_train = np.concatenate((X_down,X_up), axis=0)
y_train = np.concatenate((y_down,y_up), axis=0)
X_train = np.concatenate((X_train,X_none), axis=0)
y_train = np.concatenate((y_train,y_none), axis=0)

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


for i in data.seq(2, 7, 0.1):

    print ("----------------  Start new loop with value : "+ str(i))
    # Тренировка сети
    class_weight[1] = i
    model.fit(X_train, y_train, class_weight=class_weight, validation_split=0.05, epochs=100,
              batch_size=10, verbose=1, shuffle=True, callbacks=[checkpointer, reduce_lr, early_stopping])

    # ===================== Data load =========================

    X_down, y_down = data.get_check_data('test', 'DOWN_b38', '2D')
    X_up, y_up = data.get_check_data('test', 'UP_b38', '2D')
    X_none, y_none = data.get_check_data('test', 'NONE_b38', '2D')

    # ===================== Make prediction =====================
    y_up_pred_test = model.predict([X_up])
    y_none_pred_test = model.predict([X_none])
    y_down_pred_test = model.predict([X_down])

    # ====================== Check model =========================
    data.check_single_model(y_up_pred_test, y_none_pred_test, y_down_pred_test, sys.argv[1])





#
# y_pred_test = np.zeros(shape=(y_up.shape[0], 9))     # Сюда положим результаты прогона X_up моделями up, none, down
# for i in range(0, y_up_pred_test.shape[0]):
#     y_pred_test[i] = [y_up_pred_test[i, 0], y_up_pred_test[i, 1], y_up_pred_test[i, 2],
#                       y_none_pred_test[i, 0], y_none_pred_test[i, 1], y_none_pred_test[i, 2],
#                       y_down_pred_test[i, 0], y_down_pred_test[i, 1], y_down_pred_test[i, 2]]
#
# print("====== Save predicted data ======\n")
# np.savetxt(data.get_file_dir() + '\data\complex.csv', y_pred_test, delimiter=';')
#