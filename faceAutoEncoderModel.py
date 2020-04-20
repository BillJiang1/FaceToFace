import sys

import tensorflow as tf
import numpy as np
import cv2


# 用于图像扭曲
# class upScale(tf.keras.layers.Layer):
#     def __init__(self, output_features):
#         super(upScale, self).__init__()
#         self.conv = tf.keras.layers.Conv2D(
#             filters=output_features * 4,
#             kernel_size=3,
#             padding='same',
#             activation=tf.nn.leaky_relu
#         )
#
#     def call(self, input):
#         # print("input", input.shape)
#         x = self.conv(input)
#         x = tf.nn.depth_to_space(x, 2)
#         return x


# 定义卷积层
# class convLayer(tf.keras.layers.Conv2D):
#     def __init__(self, output_feature, k_size):
#         super(convLayer, self).__init__(
#             filters=output_feature,
#             kernel_size=k_size,
#             data_format='channels_last',
#             padding='same',
#             strides=2,
#             activation=tf.nn.leaky_relu
#         )


# class Encoder(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(
#             input_shape=(64, 64, 3),
#             filters=128,
#             kernel_size=5,
#             data_format='channels_last',
#             # strides=2,
#             padding='same',
#             activation=tf.nn.leaky_relu
#         )
#         self.conv2 = convLayer(256, 5)
#         self.conv3 = convLayer(512, 5)
#         self.conv4 = convLayer(1024, 5)
#         self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(1024 * 4 * 4, activation='relu')
#
#     def call(self, input_features):
#         # x = tf.reshape(input_features, (-1, 64, 64, 3))  # 表示暂不考虑batch_size，第一个维度暂定为1
#         # 错误示范，该行代码用于定义网络层，不应写在call方法中，会导致返回值为Symbolic tensor，即无实际数值
#         # input = tf.keras.Input(shape=(64, 64, 3))
#         x = tf.cast(input_features, dtype=tf.float32)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#
#         x = tf.keras.layers.Flatten()(x)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         x = tf.keras.layers.Reshape((4, 4, 1024))(x)
#         # print(x.shape)
#         x = upScale(512)(x)
#         return x


# class Decoder(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.upScale1 = upScale(256)
#         self.upScale2 = upScale(128)
#         self.upScale3 = upScale(64)
#         self.conv = tf.keras.layers.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')
#
#     def call(self, input):
#         x = self.upScale1(input)
#         x = self.upScale2(x)
#         x = self.upScale3(x)
#         x = self.conv(x)
#         return x


class FaceAutoEncoder:
    """
    模型说明：使用函数式API构建换脸模型，该模型由一个共用的Encoder和Decoder_init和Decoder_target组成，训练时，将图片通过共用的
            Encoder，根据图片类别分别在Decoder_init和Decoder_target上输出，训练Encoder与Decoder，进行换脸时，将initial face 通过
            Decoder_target，得到换脸后的结果
    """

    def __init__(self):
        super(FaceAutoEncoder, self).__init__()
        self.encoder = self.Encoder()
        self.decoder_init = self.Decoder()
        self.decoder_target = self.Decoder()
        # 设置输入格式为64×64的rgb图像
        input_ = tf.keras.Input((64, 64, 3))
        output_init = self.decoder_init(self.encoder(input_))
        output_target = self.decoder_target(self.encoder(input_))

        self.autoencoder_init = tf.keras.Model(input_, output_init)
        self.autoencoder_target = tf.keras.Model(input_, output_target)

        # self.autoencoder_init.summary()
        # self.autoencoder_target.summary()

    # 自定义L2损失函数,返回(batch_size,)的损失函数
    # def L2Loss(self, y_true, y_pred, batch_size):
    #     # 确保进行减法的两个tensor类型一致
    #     y_true = tf.cast(y_true, dtype=np.float32)
    #     y_pred = tf.cast(y_pred, dtype=np.float32)
    #     sub = tf.math.subtract(y_true, y_pred)
    #     result = []
    #     for i in range(batch_size):
    #         temp = sub[i]
    #         loss = tf.nn.l2_loss(temp)
    #         result.append(loss)
    #     return tf.convert_to_tensor(result)
    def upScale(self, output_features,input_):
        x = tf.keras.layers.Conv2D(
            filters=output_features * 4,
            kernel_size=3,
            padding='same',
            activation=tf.nn.leaky_relu
        )(input_)
        x = tf.nn.depth_to_space(x, 2)
        return x

    def convLayer(self, output_feature, k_size):
        layer = tf.keras.layers.Conv2D(
            filters=output_feature,
            kernel_size=k_size,
            data_format='channels_last',
            padding='same',
            strides=2,
            activation=tf.nn.leaky_relu)
        return layer

    def Encoder(self):
        input_ = tf.keras.Input(shape=(64, 64, 3))
        # x = tf.cast(input, dtype=tf.float32)
        x = self.convLayer(128, 5)(input_)
        x = self.convLayer(256, 5)(x)
        x = self.convLayer(512, 5)(x)
        x = self.convLayer(1024, 5)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(1024 * 4 * 4, activation='relu')(x)
        x = tf.keras.layers.Reshape((4, 4, 1024))(x)
        x = self.upScale(512,x)
        return tf.keras.Model(input_, x)

    def Decoder(self):
        input_ = tf.keras.Input(shape=(8, 8, 512))
        x = self.upScale(256,input_)
        x = self.upScale(128,x)
        x = self.upScale(64,x)
        x = tf.keras.layers.Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return tf.keras.Model(input_, x)

    def save_model(self):
        checkpoint_init = tf.train.Checkpoint(model=self.autoencoder_init)
        checkpoint_target = tf.train.Checkpoint(model=self.autoencoder_target)
        checkpoint_init.save("Save1/PhoebeSave/model.ckpt")
        checkpoint_target.save("Save1/AnneSave/model.ckpt")

    def restore_model(self):
        checkpoint_init = tf.train.Checkpoint(model=self.autoencoder_init)
        checkpoint_target = tf.train.Checkpoint(model=self.autoencoder_target)
        checkpoint_init.restore(tf.train.latest_checkpoint('Save1/PhoebeSave'))
        checkpoint_target.restore(tf.train.latest_checkpoint('Save1/AnneSave'))


if __name__ == '__main__':

    model = FaceAutoEncoder()
    model.restore_model()
    img = cv2.imread('face/face1.jpg')

    img1 = cv2.imread('face/face2.jpg')
    cv2.imshow("img_out", img1)
    img = cv2.resize(img,(64,64))
    img = tf.reshape(img, (-1,64, 64, 3))
    img = tf.divide(img, 255)
    img1 = tf.reshape(img1, (-1,64, 64, 3))
    img1 = tf.divide(img1, 255)
    y = model.autoencoder_target(img)
    y1 = model.autoencoder_target(img1)
    output_img = y.numpy()
    output_img = np.reshape(output_img, (64, 64, 3))
    output_img1 = y1.numpy()
    output_img1 = np.reshape(output_img1, (64, 64, 3))
    print('img', img)
    print('y:', y)
    print('y1', y1)
    cv2.imshow("img_out1", output_img1)
    cv2.waitKey()
