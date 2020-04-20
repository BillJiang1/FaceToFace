# 旨在测试普氏分析算法

import numpy as np
import dlib
import cv2
from skimage import io

cnn_face_model = "dlibModel/mmod_human_face_detector.dat"  # 基于cnn的人脸检测模型
predict_model = "dlibModel/shape_predictor_68_face_landmarks.dat"  # 用于标注人脸68个特征点

detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_model)
shape_predictor = dlib.shape_predictor(predict_model)


# points1∈R^(68*2)

def procrutes_test(points1, points2):
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
    return np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T))


if __name__ == '__main__':
    img_file = "VideoSnapshot/friends-1.jpg"
    img_file2 = "actorMenu/Joey.jpg"
    img_file3 = "actorMenu/Chandelier.jpg"
    print("Processing file:{}".format(img_file))
    img = io.imread(img_file2)
    img2 = io.imread(img_file3)
    x,y,_ = img.shape
    # 使用基于CNN的人脸检测模型检测人脸
    # dets = cnn_face_detector(img, 1)
    points = []
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        # point = np.array()
        pt_list = []
        shape = shape_predictor(img, d)  # <class 'dlib.full_object_detection'>
        for index, pt in enumerate(shape.parts()):
            temp = [pt.x, pt.y]
            # print("temp:{}".format(temp))
            pt_list.append(temp)
        pt_list = np.asarray(pt_list)
        points.append(np.mat(pt_list))
    dets = detector(img2, 1)
    for i, d in enumerate(dets):
        # point = np.array()
        pt_list = []
        shape = shape_predictor(img2, d)  # <class 'dlib.full_object_detection'>
        for index, pt in enumerate(shape.parts()):
            temp = [pt.x, pt.y]
            # print("temp:{}".format(temp))
            pt_list.append(temp)
        pt_list = np.asarray(pt_list)
        points.append(np.mat(pt_list))

    M = procrutes_test(points[1], points[0])
    img_new = cv2.warpAffine(img2, M, (y,x))
    cv2.imshow("output",img_new)
    cv2.waitKey()
