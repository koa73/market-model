import tensorflow as tf
import ComputeSum as c

x = tf.ones((2, 2))
my_sum = c.ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())