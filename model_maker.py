#!/usr/bin/env python3.5

from keras.layers import Input, Dense, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import regularizers
import datama as D

data = D.DataManager("USDRUB_TOM", 5, 1)

X_train, y_train_c = data.get_edu_data()
y_train = data.reshapy_y_by_coll(y_train_c, 1)      # Get only high

print(X_train)

# This returns a tensor
inputs = Input(shape=(20,) )

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(60, activation='relu')(inputs)
x = Dense(60, activation='relu')(x)
x = Dense(60, activation='relu')(x)
x = Dense(60, activation='relu')(x)
predictions = Dense(1,  activation='relu', name="output")(x)

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

#predict = data.denorm_y_array(model.predict(X_test))    # Предсказания
predict = model.predict(X_test)    # Предсказания

print('--------------------------------------------------------')
print(predict)
print('--------------------------------------------------------')
#print(data.denorm_y(y_test_shaped))
print(y_test_shaped)
print('========================================================')
#data.predict_report(y_test_shaped, predict)

for i in range(len(y_test_shaped)):
    print(predict[i], y_test_shaped[i], "\t", predict[i][0]-y_test_shaped[i])

# Сохраняем сеть
#data.save(model)