import numpy as np
import sys
import modelMaker as m

if (len(sys.argv) < 2):
    print("Argument not found ")
    exit(0)

data = m.ModelMaker()

# ===================== Constants =========================

data_path = data.get_file_dir()+'/data/'

# ===================== Data load =========================

X_down, y_down = data.get_check_data('test', 'DOWN_b38', '2D')

X_up, y_up = data.get_check_data('test', 'UP_b38', '2D')

X_none, y_none = data.get_check_data('test', 'NONE_b38', '2D')

# ===================== Model =========================

print("\n====== Load Model ======")
model = data.model_loader('weights_'+sys.argv[1])

print("====== Prediction ======\n")

y_up_pred_test = model.predict([X_up])
y_none_pred_test = model.predict([X_none])
y_down_pred_test = model.predict([X_down])

data.check_single_model(y_up_pred_test, y_none_pred_test, y_down_pred_test, sys.argv[1])

y_pred_test = np.zeros(shape=(y_up.shape[0], 9))     # Сюда положим результаты прогона X_up моделями up, none, down

for i in range(0, y_up_pred_test.shape[0]):
    y_pred_test[i] = [y_up_pred_test[i, 0], y_up_pred_test[i, 1], y_up_pred_test[i, 2],
                      y_none_pred_test[i, 0], y_none_pred_test[i, 1], y_none_pred_test[i, 2],
                      y_down_pred_test[i, 0], y_down_pred_test[i, 1], y_down_pred_test[i, 2]]

print("====== Save predicted data ======\n")
np.savetxt(data_path + 'complex.csv', y_pred_test, delimiter=';')

