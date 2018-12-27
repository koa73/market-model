#!/usr/bin/env python3.5

from keras.layers import Input, Dense, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
import datama_new as D

data = D.DataManager("USDRUB_TOM_2", 4, 1)

X_train, y_train_c = data.get_edu_data()
y_train = data.reshapy_y_by_coll(y_train_c)      # Get only high imp

print(y_train)


# This returns a tensor
inputs = Input(shape=(15,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(160, activation='relu')(inputs)
x = Dense(120, activation='relu')(x)
x = Dense(80, activation='relu')(x)
predictions = Dense(1, name="output")(x)

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
'''
уменьшать значение шага градиентного спуска
'''
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=0.000001, verbose=1)
model.fit(X_train, y_train, epochs=100, batch_size=1, validation_split=0.1, verbose=2, callbacks=[checkpointer,
                                                                                                  reduce_lr])  # starts training

# Тестирование модели
X_test, y_test = data.get_test_data()
y_test_shaped = data.reshapy_y_by_coll(y_test, 1)

mse, mae = model.evaluate(X_test, y_test_shaped, verbose=0)            # Проверка на тестовых данных, определяем величину ошибок
print("MSE  %f" % mse)
print("MAE  %f" % mae)

#predict = data.denorm_y_array(model.predict(X_test))    # Предсказания
predict = model.predict(X_test)    # Предсказания

X_test = data.get_test_denorm_data()
print('========================================================')

for i in range(len(y_test_shaped)-1):
    print(X_test[i][16:17], predict[i]/10*X_test[i-1][16:17])
    print(predict[i], y_test_shaped[i], "\t", predict[i][0]-y_test_shaped[i])
