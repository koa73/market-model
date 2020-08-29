import tensorflow as tf
import ShaperLayer as c

x = tf.constant([0.58, 0.14, 0.28, 0.33, 0.15, 0.52, 0.16, 0.06, 0.78])
print(x.numpy())
my_sum = c.ShaperLayer(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())