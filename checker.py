import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

import dataMiner as D
if (len(sys.argv) < 2):
    print("Argument not found ")
    exit(0)

data = D.DataMiner(3)

# ===================== Constants =========================

data_path = data.get_current_dir()+'/data/'
test_path = data_path + "test/cases/binary/"
prefix = 'b34'

# ===================== Data load =========================

with open(test_path + 'test_X_DOWN_'+prefix+'.npy', 'rb') as fin:
    X_down = np.load(fin)

with open(test_path + 'test_y_DOWN_'+prefix+'.npy', 'rb') as fin:
    y_down = np.load(fin)

print('X_down.shape: ', X_down.shape)
print('y_down.shape: ', y_down.shape)

with open(test_path + 'test_X_UP_'+prefix+'.npy', 'rb') as fin:
    X_up = np.load(fin)

with open(test_path + 'test_y_UP_'+prefix+'.npy', 'rb') as fin:
    y_up = np.load(fin)

print('X_up.shape: ', X_up.shape)
print('y_up.shape: ', y_up.shape)

with open(test_path + 'test_X_NONE_'+prefix+'.npy', 'rb') as fin:
    X_none = np.load(fin)

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


print("====== Test ======")
mse_up, mae_up = model.evaluate(x=[X_up], y=y_up)
print("MSE on UP %f" % mse_up)
print("MAE on UP %f" % mae_up)
mse_down, mae_down = model.evaluate(x=[X_down], y=y_down)
print("MSE on DOWN %f" % mse_down)
print("MAE on DOWN %f" % mae_down)
mse_none, mae_none = model.evaluate(x=[X_none], y=y_none)
print("MSE on NONE %f" % mse_none)
print("MAE on NONE %f" % mae_none)

print("====== Prediction ======\n")
y_up_pred_test = model.predict([X_up])
y_none_pred_test = model.predict([X_none])
y_down_pred_test = model.predict([X_down])

y_up_pred_test_classes = np.argmax(y_up_pred_test, axis=1)
y_up_pred_test_max_probas = np.max(y_up_pred_test, axis=1)
rounded_labels_up=np.argmax(y_up, axis=1)

y_none_pred_test_classes = np.argmax(y_none_pred_test, axis=1)
y_none_pred_test_max_probas = np.max(y_none_pred_test, axis=1)
rounded_labels_none=np.argmax(y_none, axis=1)

y_down_pred_test_classes = np.argmax(y_down_pred_test, axis=1)
y_down_pred_test_max_probas = np.max(y_down_pred_test, axis=1)
rounded_labels_down=np.argmax(y_down, axis=1)

print("Test accuracy: UP ", accuracy_score(rounded_labels_up, y_up_pred_test_classes))
print("Test accuracy: DOWN ", accuracy_score(rounded_labels_down, y_down_pred_test_classes))
print("Test accuracy: NONE ", accuracy_score(rounded_labels_none, y_none_pred_test_classes))

np.savetxt('up.csv', pred_up, delimiter=',')
np.savetxt('none.csv', pred_none, delimiter=',')
np.savetxt('down.csv', pred_down, delimiter=',')

exit(0)

fig, axs = plt.subplots(3, 1)

t = np.arange(0, y_up.shape[0], 1)
line1 = axs[0].plot(t, y_up, label='y_up', color='red')
line2 = axs[0].plot(t, pred_up, label='pred_up', color='blue')
axs[0].legend()
axs[0].grid(True)

t = np.arange(0, y_down.shape[0], 1)
line1 = axs[1].plot(t, y_down, label='y_down', color='red')
line2 = axs[1].plot(t, pred_down, label='pred_down', color='blue')
axs[1].legend()
axs[1].grid(True)

t = np.arange(0, y_none.shape[0], 1)
line1 = axs[2].plot(t, y_none, label='y_none', color='red')
line2 = axs[2].plot(t, pred_none, label='pred_none', color='blue')
axs[2].legend()
axs[2].grid(True)
