import numpy as np
#import mlflow
#import mlflow.tensorflow
import shutil
import ConcatLayer as c
import tensorflow as tf
print("We're using TF", tf.__version__)

# --- Constants ---
data_path = 'data/'
model_path = 'model/'
stats_path = 'tensorboard/'
weights_file = 'tmp_weights'

X_up_set_name = 'edu_X_b25_150.npy'
y_up_set_name = 'edu_y_b25_150.npy'

X_down_set_name = 'edu_X_b25_150.npy'
y_down_set_name = 'edu_y_b25_150.npy'

X_none_set_name = 'edu_X_b25_150.npy'
y_none_set_name = 'edu_y_b25_150.npy'

batch_size = 10
epochs = 50
learn_count = 2

# --- Load data ---

with open(data_path + X_up_set_name, 'rb') as x_up_file:
    X_up_train = np.load(x_up_file)
    X_up_train = X_up_train.reshape(X_up_train.shape[0], -1)

with open(data_path + y_up_set_name, 'rb') as y_up_file:
    y_up_train = np.load(y_up_file)

print('X_up_train.shape:', X_up_train.shape)
print('y_up_train.shape:', y_up_train.shape)

with open(data_path + X_none_set_name, 'rb') as x_none_file:
    X_none_train = np.load(x_none_file)
    X_none_train = X_none_train.reshape(X_none_train.shape[0], -1)

with open(data_path + y_none_set_name, 'rb') as y_none_file:
    y_none_train = np.load(y_none_file)

print('X_none_train.shape:', X_none_train.shape)
print('y_none_train.shape:', y_none_train.shape)

with open(data_path + X_down_set_name, 'rb') as x_down_file:
    X_down_train = np.load(x_down_file)
    X_down_train = X_down_train.reshape(X_down_train.shape[0], -1)

with open(data_path + y_down_set_name, 'rb') as y_down_file:
    y_down_train = np.load(y_down_file)

print('X_down_train.shape:', X_down_train.shape)
print('y_down_train.shape:', y_down_train.shape)

# --- Load separate model ---

print("\n --- Load previous trained model ---")

# Load network
print('Load JSON as: ', model_path + "up.json")
json_file = open(model_path + "up.json", "r")
model_up_json = json_file.read()
json_file.close()
print('Load JSON as: ', model_path + "down.json")
json_file = open(model_path + "down.json", "r")
model_down_json = json_file.read()
json_file.close()
print('Load JSON as: ', model_path + "none.json")
json_file = open(model_path + "none.json", "r")
model_none_json = json_file.read()
json_file.close()

model_up = tf.keras.models.model_from_json(model_up_json)
model_up.load_weights(model_path + "up.h5")
model_up.trainable = False
model_up.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])

model_down = tf.keras.models.model_from_json(model_down_json)
model_down.load_weights(model_path + "down.h5")
model_down.trainable = False
model_down.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])

model_none = tf.keras.models.model_from_json(model_none_json)
model_none.load_weights(model_path + "none.h5")
model_none.trainable = False
model_none.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])

print(" --- Done ---")
print("\n --- Create complex model ---")

def make_model(model_up, model_none, model_down):
    # --- COMPLEX ---
    input_layer_concat = tf.keras.layers.concatenate([model_up.output, model_none.output, model_down.output])
    #hidden_concat_custom = c.ConcatLayer()(input_layer_concat)
    #hidden_concat_dense1 = tf.keras.layers.Dense(180, name='hidden_concat_dense1_complex')(hidden_concat_custom)
    #hidden_concat_dense2 = tf.keras.layers.Dense(90, name='hidden_concat_dense2_complex')(hidden_concat_dense1)
    #hidden_concat_dense3 = tf.keras.layers.Dense(30, name='hidden_concat_dense3_complex')(hidden_concat_dense2)
    #output_complex = tf.keras.layers.Dense(3, activation='softmax', name='output_complex')(hidden_concat_dense3)
    output_complex = c.ConcatLayer()(input_layer_concat)
    #
    model = tf.keras.models.Model(inputs=[model_up.inputs, model_none.inputs, model_down.inputs], outputs=[output_complex])
    return model

model_complex = make_model(model_up, model_none, model_down)
#model_complex.trainable = False
print(model_complex.summary())
#exit(0)
print(" --- Done ---")
model_complex.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])

print('\n--- Save structure of model ---\n')
json_file = open(model_path + "complex_2.json", "w")
json_file.write(model_complex.to_json())
json_file.close()
print(" --- Done ---")

# Сохранение модели с лучшими параметрами
checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor='loss', filepath=model_path + weights_file + '_complex.h5', verbose=2, save_best_only=True)
# Уменьшение коэфф. обучения при отсутствии изменения ошибки в течении learn_count эпох
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=learn_count, min_lr=0.000001, verbose=2)
# Остановка при переобучении. patience - сколько эпох мы ждем прежде чем прерваться.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2, mode='auto')
print("\n--- Start fit of model ---\n")

model_complex.fit([X_up_train, X_none_train, X_down_train], y_up_train,
                  validation_split=0.05, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True,
                  callbacks=[checkpointer, reduce_lr, early_stopping])
print("\n--- Save weights of model ---\n")
#shutil.move(model_path + weights_file + '_complex.h5', model_path + "complex.h5")
model_complex.save(model_path + "complex_2.h5")

# ===================== Data load =========================
data_path = 'test/'
result_path = 'result/'
with open(data_path + 'test_X_UP_b38.npy', 'rb') as up_file:
    X_up = np.load(up_file)
with open(data_path + 'test_y_UP_b38.npy', 'rb') as up_file:
    y_up = np.load(up_file)
print('X_up.shape: ', X_up.shape)
print('y_up.shape: ', y_up.shape)

with open(data_path + 'test_X_NONE_b38.npy', 'rb') as none_file:
    X_none = np.load(none_file)
with open(data_path + 'test_y_NONE_b38.npy', 'rb') as none_file:
    y_none = np.load(none_file)
print('X_none.shape: ', X_none.shape)
print('y_none.shape: ', y_none.shape)

with open(data_path + 'test_X_DOWN_b38.npy', 'rb') as down_file:
    X_down = np.load(down_file)
with open(data_path + 'test_y_DOWN_b38.npy', 'rb') as down_file:
    y_down = np.load(down_file)
print('X_down.shape: ', X_down.shape)
print('y_down.shape: ', y_down.shape)

print("====== Prediction ======\n")

y_up_pred_test = np.array(model_complex.predict([X_up, X_up, X_up]))
y_none_pred_test = np.array(model_complex.predict([X_none, X_none, X_none]))
y_down_pred_test = np.array(model_complex.predict([X_down, X_down, X_down]))
print(y_up_pred_test.shape)
print(y_none_pred_test.shape)
print(y_down_pred_test.shape)

y_pred_test = np.zeros(shape=(y_up.shape[0], 9))     # Сюда положим результаты прогона X_up моделями up, none, down

for i in range(0, y_up_pred_test.shape[0]):
    y_pred_test[i] = [y_up_pred_test[i, 0], y_up_pred_test[i, 1], y_up_pred_test[i, 2],
                      y_none_pred_test[i, 0], y_none_pred_test[i, 1], y_none_pred_test[i, 2],
                      y_down_pred_test[i, 0], y_down_pred_test[i, 1], y_down_pred_test[i, 2]]

print("====== Save predicted data ======\n")
np.savetxt(result_path + 'complex_2.csv', y_pred_test, delimiter=',')

