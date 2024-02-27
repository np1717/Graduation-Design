# coding=utf-8

import cv2
import dlib


def labelImage(path, out):
    path = "test_img/test1.jpg"
    print(path)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸分类器
    detector = dlib.get_frontal_face_detector()
    # 获取人脸检测器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    dets = detector(gray, 1)
    for face in dets:
        shape = predictor(img, face)  # 寻找人脸的68个标定点
        # 遍历所有点，打印出其坐标，并圈出来
        print("\n")
        i = 0

        for pt in shape.parts():
            print(str(i) + "\t" + str(pt) + "\t" + str(pt.y))
            i = i + 1
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (0, 0, 255), 1)
        cv2.imshow("image", img)

    cv2.imwrite(out, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# path = "data/face/1.jpg"
for i in range(1, 2):
    labelImage("test1.jpg", "data/face/" + str(i) + "labeled2.jpg")
    print("\n")