import tensorflow as tf
import faceAutoEncoderModel as faem
import cv2
import os
import numpy as np

init_img_path = 'initFaces'
target_img_path = 'targetFaces'
init_dataset_path = 'datasetRecord/init_dataset'
target_dataset_path = 'datasetRecord/target_dataset'


# def make_dataset():
#     faces = []
#     for path in os.listdir(face_path):
#         path = os.path.join(face_path, path)
#         img = cv2.imread(path)
#         img = tf.divide(img, 255)
#         img2 = img3 = img4 = img
#         cv2.flip(img, 1, img2)
#         cv2.flip(img, 0, img3)
#         cv2.flip(img, -1, img4)
#         faces.append(img)
#         faces.append(img2)
#         faces.append(img3)
#         faces.append(img4)
#     # faces = np.array(faces)
#     # labels = np.array(labels)
#     ds = tf.data.Dataset.from_tensor_slices(faces)
#     return ds

# 输入人脸数据及数据集保存路径，
def save_dataset(img_path, save_path):
    with tf.io.TFRecordWriter(save_path) as writer:
        for path in os.listdir(img_path):
            img = open(os.path.join(img_path, path), 'rb').read()
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def load_dataset(dataset_path):
    raw_dataset = tf.data.TFRecordDataset(dataset_path)
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


def convert():
    pass


if __name__ == '__main__':
    model = faem.FaceAutoEncoder()
    # ds = load_dataset()
    # save_dataset(init_img_path, init_dataset_path)
    # save_dataset(target_img_path, target_dataset_path)
    # model.train(load_dataset(init_dataset_path), 'init',True)
    model.train(load_dataset(target_dataset_path), 'target',True)
