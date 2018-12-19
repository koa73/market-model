#!/usr/bin/env python3.5

from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import datama as D
import keras.backend as K


data = D.DataManager("USDRUB_TOD", 5, 1)

X_train, y_train_c = data.get_edu_data()
y_train = data.reshapy_y_by_coll(y_train_c, 1)      # Get only high

# This returns a tensor
inputs = Input(shape=(20,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(20, activation='relu')(inputs)
x = Dense(40, activation='relu')(x)
x = Dense(20, activation='relu')(x)
predictions = Dense(1, name="output")(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath=data.get_current_dir()+ "\models\weights.h5", verbose=1, save_best_only=True)
model.fit(X_train, y_train, epochs=50, batch_size=1, validation_split=0.2, verbose=2, callbacks=[checkpointer])  # starts training
#model.fit(X_train, y_train, epochs=85, batch_size=3, validation_split=0.2, verbose=2)  # starts training

# Тестирование модели
X_test, y_test = data.get_test_data()
y_test_shaped = data.reshapy_y_by_coll(y_test, 1)


mse, mae = model.evaluate(X_test, y_test_shaped, verbose=0)            # Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)

predict = data.denorm_y_array(model.predict(X_test))    # Предсказания

print('--------------------------------------------------------')
print(predict)
print('--------------------------------------------------------')
print(data.denorm_y(y_test_shaped))
print('========================================================')
#data.predict_report(y_test_shaped, predict)

for i in range(len(y_test_shaped)):
    print(predict[i], y_test_shaped[i], "\t", predict[i][0]-y_test_shaped[i])

# Сохраняем сеть
data.save(model)