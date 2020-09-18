#!/usr/bin/env python3

import tensorflow as tf
import sys
import modelMaker as d

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
model_up = data.model_loader('weights_b25_150_6.0')
model_none = data.model_loader('weights_b25_150_3.0')
model_down = data.model_loader('weights_b25_150_4.0')
