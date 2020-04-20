# 遇到的问题
1. 使用ffmpeg进行视频转图片时遇到**图片模糊**的问题：
    将
    ```
    ffmpeg -ss 00:00:06 -t 10 -i targetVideo/testVideo.mp4 -r 24 -f image2 VideoSnapshot/image-%3d.jpg`
    ```
    改为
    ```
    ffmpeg -ss 00:00:06 -t 10 -i targetVideo/testVideo.mp4 -r 24 -qscale:v 2 -f image2 VideoSnapshot/image-%3d.jpg`
    ```
    成功实现高清图片转化
2. 面对视频中存在多个面部的时候，使用Dlib进行人脸识别的准确率不高
    设置判断阈值为0.65，即当欧氏距离小于0.65时才可判断为actorMenu中的一个
3. 人脸识别中侧脸识别准确率不高：
    目前无解决办法
4. 搭建模型过程中，出现计算资源不足的问题：
    将卷积层strides属性设置为2 即
    ```
        tf.keras.layers.Conv2D(
            filters=output_feature,
            kernel_size=[k_size, k_size],
            data_format = 'channels_last',
            padding='same',
            strides=2,
            activation=tf.nn.leaky_relu
        )
    ```
    根据不同padding的H与W的计算公式：
    1) Valid padding： new_height = (H-F+1)/S
    2) Same padding: new_height = (W/S)
    将维度减少一倍，从而减少了卷积神经网络对于计算资源的需求。
5. 图像扭曲算法的选取
    deepfakes使用upScale算法，编写模型过程中通过查找相关资料使用
    `tf.nn.depth_to_space(x,2)`函数代替 (效果未知)
6. tensor无法转换为numpy类型，使用.numpy()方法报错
   ```
   AttributeError: 'Tensor' object has no attribute 'numpy'
   ```
   使用numpy.array()发现model返回的tensor为"Symbolic tensor"而不是普通的tensor，即
   无具体数据的tensor。产生此tensor的原因为：
   将`input = tf.keras.Input(shape=(64, 64, 3))`写在了Encoder与Decoder的call方法中
   该语句定义的是形式输入，用于定义网络层而不是调用网络层
7. 如何将decoder输出的数组转化为可显示的图像：
    将数组维度转化为(W,H,Channel)，与网上查阅的资料不同，目前无须将数组元素转化为np.uint8也可显示图像
    ```
    y = model(input)
    output_img = y.numpy()
    output_img = np.reshape(output_img,(64,64,3))
    print(output_img)
    cv2.imshow("img_out",output_img)
    cv2.waitKey()
    ```
8. 关于模型的保存与恢复：
    由于本项目涉及两个AutoEncoder模型，与手册中有所出入，暂不确定是否可以将模型参数信息保存在同一个文件夹中
。观察发现，一个文件夹中只有一个checkpoint文件，其中的checkpoint_path为后保存的模型路径。
    **解决方法：** 采用子文件夹的形式分别存储
9. 经过训练，100次epochs后，误差收敛到一个较大的数值，且输入不同的图片，经过模型后的输出是相同的。
    **解决方法：** 重构模型后该问题解决
10. 作为测试的数据集均为Anne的人脸图像，训练后的Autoencoder能够较为清晰的解析源数据集中的人脸，
但是对于数据集以外的任何人脸都不具备解析能力。
    构想的解决方法：多样化数据集内容，在数据集中加入其他人脸。
    