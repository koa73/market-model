#!/usr/bin/env python3.5

from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import datama as D

data = D.DataManager("USDRUB_TOM", 4, 1)

X_train, y_train_c = data.get_edu_data()
y_train = data.reshapy_y_by_coll(y_train_c, 1)      # Get only high


# This returns a tensor
inputs = Input(shape=(15,))

# a layer instance is callable on a tensor, and returns a tensor
x1 = Dense(394, activation='relu')(inputs)
x2 = Dense(394, activation='relu')(x1)
x3 = Dense(394, activation='relu')(x2)
x4 = Dense(394, activation='relu')(x3)
predictions = Dense(1, name="output")(x4)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
data.save_conf(model)                                                  # Запись конфигурации скти для прерывания расчета
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath=data.get_current_dir()+ "\models\weights.h5", verbose=1, save_best_only=True)
model.fit(X_train, y_train, epochs=100, batch_size=1, validation_split=0.1, verbose=2, callbacks=[checkpointer])  # starts training

# Тестирование модели
X_test, y_test = data.get_test_data()
y_test_shaped = data.reshapy_y_by_coll(y_test, 1)


mse, mae = model.evaluate(X_test, y_test_shaped, verbose=0)            # Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)

predict = model.predict(X_test)    # Предсказания

print('--------------------------------------------------------')
print(predict)
print('--------------------------------------------------------')
print(y_test_shaped)
print('========================================================')
#data.predict_report(y_test_shaped, predict)

for i in range(len(y_test_shaped)):
    print(predict[i], y_test_shaped[i], "\t", predict[i][0]-y_test_shaped[i])

# Сохраняем сеть
#data.save(model)