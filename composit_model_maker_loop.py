#!/usr/bin/env python3

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


def calculate_model(model_up, model_none, model_down, idx, comment):
    # ===================== Build model =============================

    print("------------------- Build -------")
    input_layer_1 = tf.keras.layers.concatenate([model_up.output, model_none.output, model_down.output])
    output = c.ConcatLayer()(input_layer_1)
    model = tf.keras.models.Model(inputs=[model_up.inputs, model_none.inputs, model_down.inputs], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    print(model.summary())
    data.save_conf(model, 'composite_'+str(idx))  # Запись конфигурации скти для прерывания расчета

    #model.fit([X_up, X_none, X_down], y_down, validation_split=0.05, epochs=2, batch_size=10, verbose=1)
    model.save(data.get_file_dir() + "/data/model_test/weights_composite_"+str(idx)+".h5")

    # ===================== Make prediction =====================
    y_up_pred_test = model.predict([X_up, X_up, X_up])
    y_none_pred_test = model.predict([X_none, X_none, X_none])
    y_down_pred_test = model.predict([X_down, X_down, X_down])

    # ===================== Model checker =======================

    data.check_single_model(y_up_pred_test, y_none_pred_test, y_down_pred_test, 'composite_'+str(idx), comment)


i = 10
#for up in (list(range(19, 28))+list(range(34, 53))):
for up in (list(range(52, 57))):
    for none in range(120, 135):
        for down in range(63, 120):
            # ====================== Load static models =====================
            print('>>>>>> Check models UP: '+str(up)+', None: '+str(none)+', Down: '+str(down))
            model_u = data.model_loader('weights_b25_150_'+ str(up))
            model_n = data.model_loader('weights_b25_150_' + str(none))
            model_d = data.model_loader('weights_b25_150_' + str(down))
            calculate_model(model_u, model_n, model_d, i, str(up)+', '+str(none)+', '+str(down))
            i += 1



