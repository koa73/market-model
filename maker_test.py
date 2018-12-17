#!/usr/bin/env python3.5

from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import datama_new as d

data = d.DataManager("USDRUB", 5, 1)

X_train, y_train_c = data.get_edu_data()
y_train = data.reshapy_y_by_coll(y_train_c, 1)      # Get only high

print(y_train)

# This returns a tensor
inputs = Input(shape=(20,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(60, activation='relu')(inputs)
predictions = Dense(1, activation='relu', name="output")(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath=data.get_current_dir()+ "\models\weights.hdf5", verbose=1, save_best_only=True)

model.fit(X_train, y_train, epochs=10, batch_size=5, validation_split=0.01, verbose=2, callbacks=[checkpointer])  # starts training

# Тестирование модели
X_test, y_test = data.get_test_data()
y_test_shaped = data.reshapy_y_by_coll(y_test, 1)


mse, mae = model.evaluate(X_test, y_test_shaped, verbose=0, batch_size=10)            # Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)

predict = model.predict(X_test)    # Предсказания

#data.predict_report(y_test_shaped, predict)

for i in range(len(y_test_shaped)):
    print(predict[i], y_test_shaped[i], "\t", predict[i][0]-y_test_shaped[i])

# Сохраняем сеть
#data.save(model)