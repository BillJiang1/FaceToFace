# 使用Dlib进行人脸识别以及人脸特征点检测
import os
import dlib
from skimage import io
import cv2
import numpy as np

# TODO: 人脸识别准确度不高，shape_predictor参数类型不匹配

cnn_face_model = "dlibModel/mmod_human_face_detector.dat"  # 基于cnn的人脸检测模型
predict_model = "dlibModel/shape_predictor_68_face_landmarks.dat"  # 用于标注人脸68个特征点
face_rec_model = "dlibModel/dlib_face_recognition_resnet_model_v1.dat"  # dlib人脸识别模型

cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_model)
face_detector = dlib.get_frontal_face_detector()  # 基于经典HOG的人脸检测模型，在使用CPU进行计算时拥有更高的计算效率
shape_predictor = dlib.shape_predictor(predict_model)
face_recognizer = dlib.face_recognition_model_v1(face_rec_model)

menu_path = 'actorMenu'
descriptors = []
known_names = ['Chandelier', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross']


# 计算人脸识别参考的关键特征点
def get_menu_descriptors():
    for filename in os.listdir(menu_path):
        # 提取文件名中的Actor name
        actor_name = filename[:-4]
        print("Processing image:{}...".format(filename))
        img = cv2.imread(os.path.join(menu_path, filename))
        # img = io.imread(os.path.join(menu_path, filename))
        # img = dlib.load_rgb_image(filename)
        dets = face_detector(img, 1)
        print("Number of targetFaces detected:{}".format(len(dets)))

        for i, d in enumerate(dets):
            # 计算关键特征点
            shape = shape_predictor(img, d)
            face_descriptor = face_recognizer.compute_face_descriptor(img, shape)
            v = np.array(face_descriptor)
            descriptors.append(v)
            np.save("actorDescriptor/record_{}".format(actor_name), v)


# 基于经典HOG的人脸识别模型,返回识别结果
def face_recognition(img_file, target_character):
    print("Processing file:{}".format(img_file))
    img = cv2.imread(img_file)
    # 将cv2读取的bgr模式的图片转化为rgb模式
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    # img = io.imread(img_file)
    # 使用基于HOG的人脸检测模型检测人脸
    dets = face_detector(img, 1)
    # 使用基于CNN的人脸识别模型
    # dets = cnn_face_detector(img,1)
    print("Number of targetFaces detected : {}".format(len(dets)))
    top = bottom = left = right = -1
    for i, d in enumerate(dets):
        dist = []
        # 提取描述子
        shape = shape_predictor(img, d)
        face_descriptor = face_recognizer.compute_face_descriptor(img, shape)
        d_test = np.array(face_descriptor)
        # 计算欧氏距离
        for j in descriptors:
            dist_ = np.linalg.norm(j - d_test)
            dist.append(dist_)
        # 与namelist合并，排序选取最小距离
        c_d = zip(known_names, dist)
        cd_sorted = sorted(c_d, key=lambda x: x[1])
        # 使用切片裁剪人脸
        # targetFaces = img[d.top():d.bottom(), d.left():d.right()]

        # 设置人脸检测阈值，即误差小于0.6时才可确定目标身份
        if cd_sorted[0][1] <= 0.65:
            # print("The person {} is {},distance is {}".format(i, cd_sorted[0][0], cd_sorted[0][1]))
            if cd_sorted[0][0] == target_character:
                top = d.top()
                bottom = d.bottom()
                left = d.left()
                right = d.right()
        else:
            # print("The person {} is not sure,distance is {}".format(i, cd_sorted[0][1]))
            pass
    # 不能确定人物身份或图中无targetCharacter
    return top, bottom, left, right


# 基于Procrustes Analysis 进行人脸对齐,返回仿射变换矩阵
# 传入2个68×2的特征点位置矩阵，points1为待校准，points2为基准
def procurstes_analysis(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    # 消除平移T影响
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    # 消除缩放系数s的影响
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    temp = points1.T * points2
    # U，V为标准正交基，S为对矩阵的奇异值分解
    U, S, V = np.linalg.svd(temp)
    R = (U * V).T
    # print(np.hstack((((s2 / s1) * R), c2.T - (s2 / s1) * R * c1.T)))
    return np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T))


# M为仿射变换矩阵，用于将img进行校准
def face_aligment(img, M, x, y):
    img_new = cv2.warpAffine(img, M, (y, x))
    return img_new

if __name__ == '__main__':
    imgfile = "videoSnapshot/image-004.jpg"
    # if os.listdir("actorDescriptor"):
    #     for root, dirs, files in os.walk("actorDescriptor"):
    #         for f in files:
    #             descriptor_array = np.load(os.path.join("actorDescriptor", f))
    #             descriptors.append(descriptor_array)
    # else:
    #     get_menu_descriptors()
    # get_menu_descriptors()
    face_recognition(imgfile, 'Phoebe')
