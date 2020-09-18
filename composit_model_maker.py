#!/usr/bin/env python3

import tensorflow as tf
import sys
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
model_up = data.model_loader('weights_b25_150_6.0') #######!!!!!!!
model_none = data.model_loader('weights_b25_150_3.0')
model_down = data.model_loader('weights_b25_150_4.0')

# ===================== Build model =============================


input_layer_1 = tf.keras.layers.concatenate([model_up.output, model_none.output, model_down.output])
output = c.ConcatLayer()(input_layer_1)
model = tf.keras.models.Model(inputs=[model_up.inputs, model_none.inputs, model_down.inputs], outputs=[output])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
print(model.summary())
data.save_conf(model,'composite')                                                  # Запись конфигурации скти для прерывания расчета

#model = tf.keras.models.Model(inputs=[input_layer_1], outputs=[output])
