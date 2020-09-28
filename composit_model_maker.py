#!/usr/bin/env python3

import tensorflow as tf
import sys
import numpy as np
import modelMaker as d
import ConcatLayer as c

data = d.ModelMaker()

print("Start composit model making ....")

if (len(sys.argv) < 2):
    print("Argument not found ")
    exit(0)

# ===================== Case Data load =========================

X_down, y_down = data.get_check_data('test', 'DOWN_b38', '2D')
X_up, y_up = data.get_check_data('test', 'UP_b38', '2D')
X_none, y_none = data.get_check_data('test', 'NONE_b38', '2D')

# ====================== Load static models =====================
model_up = data.model_loader('weights_b25_150_0.0')  # 3.18 х 0.271
model_none = data.model_loader('weights_b25_150_126') # 2.1 х 0.263
model_down = data.model_loader('weights_b25_150_80') # 3.5 х 0.295

# ===================== Build model =============================

print("------------------- Build -------")
input_layer_1 = tf.keras.layers.concatenate([model_up.output, model_none.output, model_down.output])
output = c.ConcatLayer()(input_layer_1)
model = tf.keras.models.Model(inputs=[model_up.inputs, model_none.inputs, model_down.inputs], outputs=[output])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
print(model.summary())
data.save_conf(model,'composite_1')                                                  # Запись конфигурации скти для прерывания расчета

#model.fit([X_up, X_none, X_down], y_down, validation_split=0.05, epochs=2, batch_size=10, verbose=1)
model.save(data.get_file_dir() + "/data/model_test/weights_composite_1.h5")

# ===================== Make prediction =====================
y_up_pred_test = model.predict([X_up, X_up, X_up])
y_none_pred_test = model.predict([X_none, X_none, X_none])
y_down_pred_test = model.predict([X_down, X_down, X_down])

# ===================== Model checker =======================

data.check_single_model(y_up_pred_test, y_none_pred_test, y_down_pred_test, 'composite_1', '0,126,80')

y_pred_test = np.zeros(shape=(y_up.shape[0], 9))     # Сюда положим результаты прогона X_up моделями up, none, down

for i in range(0, y_up_pred_test.shape[0]):
    y_pred_test[i] = [y_up_pred_test[i, 0], y_up_pred_test[i, 1], y_up_pred_test[i, 2],
                      y_none_pred_test[i, 0], y_none_pred_test[i, 1], y_none_pred_test[i, 2],
                      y_down_pred_test[i, 0], y_down_pred_test[i, 1], y_down_pred_test[i, 2]]

print("====== Save predicted data ======\n")
np.savetxt(data.get_file_dir() + '/data/complex_1.csv', y_pred_test, delimiter=';')
