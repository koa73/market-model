import tensorflow as tf
import numpy as np
import sys

import dataMaker as d
if (len(sys.argv) < 2):
    print("Argument not found ")
    exit(0)

data = d.DataMaker()

# ===================== Constants =========================

data_path = data.get_file_dir()+'/data/'
test_path = data_path + "test/cases/binary/"
prefix = 'b38'

# ===================== Data load =========================

with open(test_path + 'test_X_DOWN_'+prefix+'.npy', 'rb') as fin:
    X_down = np.load(fin)
    X_down = X_down.reshape(X_down.shape[0], -1)

with open(test_path + 'test_y_DOWN_'+prefix+'.npy', 'rb') as fin:
    y_down = np.load(fin)

print('X_down.shape: ', X_down.shape)
print('y_down.shape: ', y_down.shape)

with open(test_path + 'test_X_UP_'+prefix+'.npy', 'rb') as fin:
    X_up = np.load(fin)
    X_up = X_up.reshape(X_up.shape[0], -1)

with open(test_path + 'test_y_UP_'+prefix+'.npy', 'rb') as fin:
    y_up = np.load(fin)

print('X_up.shape: ', X_up.shape)
print('y_up.shape: ', y_up.shape)

with open(test_path + 'test_X_NONE_'+prefix+'.npy', 'rb') as fin:
    X_none = np.load(fin)
    X_none = X_none.reshape(X_none.shape[0], -1)

with open(test_path + 'test_y_NONE_'+prefix+'.npy', 'rb') as fin:
    y_none = np.load(fin)

print('X_none.shape: ', X_none.shape)
print('y_none.shape: ', y_none.shape)

# ===================== Model =========================

print("\n====== Load Model ======")

# Load network
print('Load JSON as: ', data_path +"model_test/weights_"+ sys.argv[1] + ".json")
json_file = open(data_path +"model_test/weights_"+ sys.argv[1] + ".json", "r")
model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(model_json)
model.load_weights(data_path +"/model_test/weights_"+ sys.argv[1]  + ".h5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("====== Prediction ======\n")
y_up_pred_test = model.predict([X_up])
y_none_pred_test = model.predict([X_none])
y_down_pred_test = model.predict([X_down])

print(y_up_pred_test.shape)
print(y_none_pred_test.shape)
print(y_down_pred_test.shape)

y_pred_test = np.zeros(shape=(y_up.shape[0], 9))     # Сюда положим результаты прогона X_up моделями up, none, down

for i in range(0, y_up_pred_test.shape[0]):
    y_pred_test[i] = [y_up_pred_test[i, 0], y_up_pred_test[i, 1], y_up_pred_test[i, 2],
                      y_none_pred_test[i, 0], y_none_pred_test[i, 1], y_none_pred_test[i, 2],
                      y_down_pred_test[i, 0], y_down_pred_test[i, 1], y_down_pred_test[i, 2]]

print("====== Save predicted data ======\n")
np.savetxt(data_path + 'complex.csv', y_pred_test, delimiter=',')

