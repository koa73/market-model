import dataman as d
import tensorflow as tf

data = d.DataManager("USDRUB-w.csv", 5, 1)

X_train, y_train = data.get_edu_data()

print(data.get_variations_num())

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(230, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.LSTM(65, activation=tf.nn.relu, dropout=0.2))
model.add(tf.keras.layers.Dense(2))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(X_train, y_train, epochs=7, batch_size=10)      #Тренировка сети

# Сохраняем сеть
data.save(model)



