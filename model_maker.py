#!/usr/bin/env python3

from keras.layers import Input, Dense, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
import sys
import dataMiner as D

data = D.DataMiner(3)

print("Start model making ....")

X_train = data.get_edu('X_edu', '_last_'+sys.argv[0])
X_train = X_train.reshape(X_train.shape[0],-1)
y_train =  data.get_edu('y_edu', '_last_'+sys.argv[0])

print("X_edu : " + str(X_train.shape))
print("y_edu : " + str(y_train.shape))

print(X_train[0])
#input("Wait any key ....")
# This returns a tensor
inputs = Input(shape=(12,) )

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(12, activation='tanh')(inputs)
x = Dense(24, activation='tanh')(x)
x = Dense(48, activation='tanh')(x)
x = Dense(24, activation='tanh')(x)
x = Dense(6, activation='tanh')(x)
predictions = Dense(3,  activation='softmax', name="output")(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
data.save_conf(model, sys.argv[0])                                                  # Запись конфигурации скти для прерывания расчета
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath = data.get_current_dir()+ "/data/model_test/weights_"+sys.argv[0]+".h5", verbose=1, save_best_only=True)

'''
уменьшать значение шага градиентного спуска
'''
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=0.000001, verbose=1)
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.05, verbose=1, callbacks=[checkpointer,reduce_lr])  # starts training