# 测试
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

# 读取一张图片
image = cv2.imread("test.jpg")

# 调用人脸检测器
detector = dlib.get_frontal_face_detector()

# 加载人脸关键点的模型
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 灰度转化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 人脸检测
faces = detector(gray, 1)
# 循环，遍历每一张人脸，绘制人脸矩形框和关键点
for face in faces:  # (x,y,w,h)
    # 绘制矩形框
    cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
    # 预测关键点
    shape = predictor(image, face)
    # 获取到关键点的坐标
    for pt in shape.parts():
        # 获取横纵坐标
        pt_position = (pt.x, pt.y)
        # 显示/绘制关键点
        cv2.circle(image, pt_position, 3, (0, 0, 255), -1)

# 显示整个效果图
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_RGB)
plt.axis("off")
plt.show()