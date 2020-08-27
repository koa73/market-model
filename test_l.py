import tensorflow as tf
import ComputeSum as c

x = tf.constant([[1., 2.], [3., 4.]])
print(x.numpy())
my_sum = c.ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())