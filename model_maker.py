#!/usr/bin/env python3

from keras.layers import Input, Dense, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import regularizers
import dataMiner as D

data = D.DataMiner(3)

print("Start model making ....")

X_train = data.get_edu('X_edu', '_last_b3')
X_train = X_train.reshape(X_train.shape[0],-1)
y_train =  data.get_edu('y_edu', '_last_b3')

print("X_edu : " + str(X_train.shape))
print("y_edu : " + str(y_train.shape))

print(X_train[0])
#input("Wait any key ....")
# This returns a tensor
inputs = Input(shape=(12,) )

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(60, activation='tanh')(inputs)
x = Dense(60, activation='tanh')(x)
x = Dense(60, activation='tanh')(x)
x = Dense(60, activation='tanh')(x)
predictions = Dense(3,  activation='softmax', name="output")(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
#data.save_conf(model)                                                  # Запись конфигурации скти для прерывания расчета
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath=data.get_current_dir()+ "\data\model_test\weights.h5", verbose=1, save_best_only=True)
model.fit(X_train, y_train, epochs=100, batch_size=1, validation_split=0.1, verbose=1, callbacks=[checkpointer])  # starts training
