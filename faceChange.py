import tensorflow as tf
import faceAutoEncoderModel as faem
import cv2
import os
import numpy as np

batch_size = 1
epoches = 150
face_path = 'face'
img_dataset_path = 'datasetRecord/dataset'
tensorboard_path = 'tensorboard'
checkpoints_save_path = 'SaveTest'


def make_dataset():
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
    # faces = np.array(faces)
    # labels = np.array(labels)
    ds = tf.data.Dataset.from_tensor_slices(faces)
    return ds


def save_dataset():
    with tf.io.TFRecordWriter(img_dataset_path) as writer:
        for path in os.listdir(face_path):
            img = open(os.path.join(face_path, path), 'rb').read()
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def load_dataset():
    raw_dataset = tf.data.TFRecordDataset(img_dataset_path)
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_example(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])  # 解码JPEG图片
        img = tf.divide(feature_dict['image'], 255)
        return img

    dataset = raw_dataset.map(_parse_example)
    return dataset


def train(model_, ds, load_data=False):
    # ds = make_dataset()
    ds = ds.batch(batch_size)
    model = model_
    # 使用tensorboard将训练过程可视化
    summary_writer = tf.summary.create_file_writer(tensorboard_path)
    if load_data and (os.listdir(checkpoints_save_path)):
        model.restore_model()
    for epoch in range(epoches):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for img in ds:
            batch_loss = train_per_step(model, img, "init")
            epoch_loss_avg(batch_loss)
        with summary_writer.as_default():  # 希望使用的记录器
            tf.summary.scalar("loss", batch_loss, step=epoch)
        print("Epoch {} Loss:{}".format(epoch, epoch_loss_avg.result()))
        if epoch % 10 == 0:
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
    ds = load_dataset()
    train(faceAutoEncoder, ds)
