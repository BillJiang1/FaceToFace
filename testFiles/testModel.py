import faceAutoEncoderModel as faem
import cv2
import tensorflow as tf
import numpy as np

img = cv2.imread('../targetFaces/face0.jpg')
img = cv2.resize(img,(64,64))
img = tf.convert_to_tensor(img)

model = faem.FaceAutoEncoder()
y = model(img)
print(y.numpy())
# loss = tf.losses.mean_absolute_error(y_true = img,y_pred = y)
img = tf.cast(img,dtype = np.float32)
result = tf.math.subtract(img,y)
print(img.numpy())

print(result)
loss = tf.nn.l2_loss(result)
print(loss.numpy())