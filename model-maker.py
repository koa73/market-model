import dataman as d
import tensorflow as tf

data = d.DataManager("USDRUB.csv", 5, 1)

X_train, y_train = data.get_edu_data()

print(data.get_variations_num())

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(180, input_shape=(X_train.shape[1], X_train.shape[2]),recurrent_dropout=0.2, dropout=0.2))
#model.add(tf.keras.layers.LSTM(128, activation=tf.nn.tanh, recurrent_dropout=0.2, dropout=0.2, return_sequences=True))
model.add(tf.keras.layers.Dense(90, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(2))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, batch_size=10)      #Тренировка сети

# Сохраняем сеть
data.save(model)



