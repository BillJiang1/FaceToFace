import tensorflow as tf

A = tf.constant([[1., 2.], [3., 4.]])
print(type(A))
print(A.numpy())
print('Is GPU available:')
print(tf.test.is_gpu_available())
print('Is the Tensor on gpu #0:')
print(A.device)