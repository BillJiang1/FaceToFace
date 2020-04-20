import tensorflow as tf
import faceAutoEncoderModel as faem
import cv2
import os
import numpy as np

batch_size = 10
epoches = 150


def make_dataset():
    face_path = 'face'
    faces = []
    for path in os.listdir(face_path):
        path = os.path.join(face_path, path)
        img = cv2.imread(path)
        img = tf.divide(img, 255)
        img2 = img3 = img4 = img
        cv2.flip(img, 1, img2)
        cv2.flip(img, 0, img3)
        cv2.flip(img, -1, img4)
        faces.append(img)
        faces.append(img2)
        faces.append(img3)
        faces.append(img4)
    ds = tf.data.Dataset.from_tensor_slices(faces)
    return ds


def train(model_, load_data=0):
    ds = make_dataset()
    ds = ds.batch(batch_size)
    model = model_
    # 使用tensorboard将训练过程可视化
    summary_writer = tf.summary.create_file_writer('tensorboard')
    if load_data == 1 and (os.listdir('Save')):
        model.restore_model()
    for epoche in range(epoches):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for img in ds:
            batch_loss = train_per_step(model, img, 'target')
            epoch_loss_avg(batch_loss)
        with summary_writer.as_default():  # 希望使用的记录器
            tf.summary.scalar("loss", batch_loss, step=epoche)
        print("Epoch {} Loss:{}".format(epoche, epoch_loss_avg.result()))
        if epoche % 10 == 0:
            model.save_model()
    model.save_model()


# @tf.function 使用该语句修饰会报错 ValueError: tf.function-decorated function tried to create variables on non-first call.
def train_per_step(model, img, label):
    my_optimizer = tf.optimizers.Adam()
    faceAutoEncoder = model
    if label == 'init':
        model = faceAutoEncoder.autoencoder_init
    elif label == 'target':
        model = faceAutoEncoder.autoencoder_target
    else:
        print('Label Error！')
        return
    with tf.GradientTape() as tape:
        predicts = model(img)
        loss = tf.losses.mean_absolute_error(img, predicts)
        # loss = faceAutoEncoder.L2Loss(img, predicts, batch_size)
        loss = tf.reduce_mean(loss)
        print(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    grand_and_vars = zip(gradients, model.trainable_variables)
    my_optimizer.apply_gradients(grand_and_vars)
    return loss


def convert():
    pass


if __name__ == '__main__':
    faceAutoEncoder = faem.FaceAutoEncoder()
    train(faceAutoEncoder)
