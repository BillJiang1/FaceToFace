import cv2
import dlib
import os
import tensorflow as tf

imgs_path = '..\Anne imgs'
def test():

    i = 0
    for path in os.listdir('../Anne imgs'):
        img_path = os.path.join(imgs_path, path)
        face_detector = dlib.get_frontal_face_detector()
        img = cv2.imread(img_path)
        dets = face_detector(img, 1)
        print("processing file {}".format(img_path))
        for j, d in enumerate(dets):
            top = d.top()
            bottom = d.bottom()
            right = d.right()
            left = d.left()
            face = img[top:bottom, left:right]
            # 将面部图片规定为64×64大小
            face = cv2.resize(face, (64,64), interpolation=cv2.INTER_AREA)
            cv2.imwrite("face{}.jpg".format(i), face)
        i += 1

if __name__ == '__main__':
    img = cv2.imread("../Anne imgs/250.jpg")
    cv2.imshow("img1",img)
    cv2.waitKey()
    print(img)
    img = tf.divide(img,255)
    print(img)
    cv2.imshow("img2",img)
    cv2.waitKey()
