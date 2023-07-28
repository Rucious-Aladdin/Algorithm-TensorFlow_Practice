import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")
print(hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)

print(c)
#########