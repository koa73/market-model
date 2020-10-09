#!/usr/bin/env python3
import time
import tensorflow as tf
import sys
import numpy as np
import modelMaker as d
import ConcatLayer as c

data = d.ModelMaker()

print("Start composit model making ....")

# ===================== Case Data load =========================

X_down, y_down = data.get_check_data('test', 'DOWN_b38', '2D')
X_up, y_up = data.get_check_data('test', 'UP_b38', '2D')
X_none, y_none = data.get_check_data('test', 'NONE_b38', '2D')

x1 = [52, 52, 52, 56, 56, 54, 54, 54, 54, 56, 56, 56, 54, 56, 56, 54, 54, 55, 56, 55, 56, 55, 53, 54, 54, 54, 53, 55, 56, 54, 54, 53, 54, 55, 53, 56, 53, 53, 55, 55, 56, 55, 54, 53, 55, 53, 54, 53, 52, 53]
x2 = [133, 122, 123, 121, 128, 128, 127, 132, 121, 121, 130, 128, 127, 127, 133, 125, 123, 133, 125, 121, 127, 128, 129, 123, 125, 133, 121, 127, 122, 131, 122, 132, 122, 128, 128, 123, 128, 121, 122, 133, 134, 123, 133, 121, 123, 127, 121, 127, 128, 133]
x3 = [70, 70, 70, 70, 70, 70, 70, 70, 68, 69, 70, 69, 69, 70, 70, 70, 69, 70, 70, 70, 69, 70, 70, 70, 69, 70, 70, 70, 70, 70, 70, 70, 69, 69, 69, 70, 68, 69, 70, 69, 70, 69, 69, 68, 70, 68, 73, 69, 74, 70]

def calculate_model(model_up, model_none, model_down, idx, comment):
    # ===================== Build model =============================

    sec = str(time.time())
    print("------------------- Build -------")
    input_layer_1 = tf.keras.layers.concatenate([model_up.output, model_none.output, model_down.output], name='1' + sec)
    output = c.ConcatLayer()(input_layer_1)
    model = tf.keras.models.Model(inputs=[model_up.inputs, model_none.inputs, model_down.inputs], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    print(model.summary())
    data.save_conf(model, 'composite_'+str(idx))  # Запись конфигурации сети для прерывания расчета

    model.save(data.get_file_dir() + "/data/model_test/weights_composite_"+str(idx)+".h5")

    # ===================== Make prediction =====================
    y_up_pred_test = model.predict([X_up, X_up, X_up])
    y_none_pred_test = model.predict([X_none, X_none, X_none])
    y_down_pred_test = model.predict([X_down, X_down, X_down])

    # ===================== Model checker =======================

    data.check_single_model(y_up_pred_test, y_none_pred_test, y_down_pred_test, 'composite_'+str(idx), comment)


i = 10
#for up in (list(range(0, 28))+list(range(34, 53))):
#for up in (list(range(18, 19))):
    #for none in range(120, 135):
     #   for down in range(63, 120):


for x in range (len(x1)):
    # ====================== Load static models =====================
    print('>>>>>> Check models UP: ' + str(x1[x]) + ', None: ' + str(x2[x]) + ', Down: ' + str(x3[x]))
    model_u = data.model_loader('weights_b25_150_' + str(x1[x]))
    model_n = data.model_loader('weights_b25_150_' + str(x2[x]))
    model_d = data.model_loader('weights_b25_150_' + str(x3[x]))
    calculate_model(model_u, model_n, model_d, i, str(x1[x]) + ', ' + str(x2[x]) + ', ' + str(x3[x]))
    i += 1
model_u = data.model_loader('weights_composite_10_135')
model_n = data.model_loader('weights_composite_18_143')
model_d = data.model_loader('weights_composite_21_146')
calculate_model(model_u, model_n, model_d, i, '10_135_18_143_21_146')