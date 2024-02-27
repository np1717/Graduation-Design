# 把数据集按照68点分类
import os
import cv2
from tqdm import tqdm
import dlib
import math
import numpy as np

# 获取人脸检测器
detector = dlib.get_frontal_face_detector()

# 获取人脸关键点检测器、
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 指定图像文件夹路径
image_folder = "image"

# 创建 0-6 的文件夹
for i in range(7):
    os.makedirs(str(i), exist_ok=True)

# 获取图像文件夹中的所有文件名
image_files = os.listdir(image_folder)

# 使用 tqdm 显示进度条
for image_file in tqdm(image_files, desc="Moving images"):
    # 构造图像文件的完整路径
    image_path = os.path.join(image_folder, image_file)

    # 读取图像文件
    image = cv2.imread(image_path)

    # 灰度转化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = detector(gray, 1)
    # 绘制人脸的矩形框和关键点
    for face in faces:
        # 检测关键点
        shape = predictor(gray, face)
        landmarks = predictor(gray, face)  # 寻找人脸的68个标定点
        # 获取各点坐标坐标
        x48, y48 = landmarks.part(48).x, landmarks.part(48).y
        x66, y66 = landmarks.part(66).x, landmarks.part(66).y
        x54, y54 = landmarks.part(54).x, landmarks.part(54).y
        x37, y37 = landmarks.part(37).x, landmarks.part(37).y
        x38, y38 = landmarks.part(38).x, landmarks.part(38).y
        x41, y41 = landmarks.part(41).x, landmarks.part(41).y
        x40, y40 = landmarks.part(40).x, landmarks.part(40).y
        x36, y36 = landmarks.part(36).x, landmarks.part(36).y
        x39, y39 = landmarks.part(39).x, landmarks.part(39).y
        x17, y17 = landmarks.part(17).x, landmarks.part(17).y
        x19, y19 = landmarks.part(19).x, landmarks.part(19).y
        x21, y21 = landmarks.part(21).x, landmarks.part(21).y
        x62, y62 = landmarks.part(62).x, landmarks.part(62).y

        # 获取最高点和最低点的坐标
        highest_point = min(landmarks.part(i).y for i in range(0, 68))
        lowest_point = max(landmarks.part(i).y for i in range(0, 68))

        # 计算48点与66点连线与水平线的夹角（弧度）
        angle_48_66 = math.atan2(y66 - y48, x66 - x48)
        # 将弧度转换为角度a_left
        a_left = math.degrees(angle_48_66)

        # 计算54点与66点连线与水平线的夹角（弧度）
        angle_54_66 = math.atan2(y66 - y54, x66 - x54)
        # 将弧度转换为角度 a_right
        a_right = math.degrees(angle_54_66)

        # 计算眼高和眼宽
        eye_height = abs(y38 - y40)
        eye_width = abs(x39 - x36)
        # 计算眼高与眼宽的比值b
        b = eye_height / eye_width

        # 计算点19到点17和点21连线的向量
        vec1 = np.array([x17 - x19, y17 - y19])
        vec2 = np.array([x21 - x19, y21 - y19])
        # 计算夹角
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        theta = np.arccos(cos_theta)
        # 将弧度转换为角度c
        c = math.degrees(theta)

        # 计算脸的长度
        face_length = lowest_point - highest_point
        # 计算66点与62点的纵坐标差值占脸长的比例L
        L = abs(y66 - y62) / face_length

        # 判断表情类型并输出
        output = 'Else'

        if a_left > 0 and a_right > 0 and b > 0.25 and c > 120:
            if L > 0.03:
                output = 'Happy'  # 高兴
            else:
                output = 'Understand'  # 理解
        if a_left > 0 > a_right or a_left < 0 < a_right:
            if b > 0.25 and c > 120:
                output = 'Listen'  # 聆听
            else:
                output = 'Despise'  # 轻视
        if a_left < 0 and a_right < 0:
            if c < 120:
                output = 'Puzzled'  # 困惑
            else:
                if b > 0.25:
                    output = 'Listen'  # 聆听
                else:
                    output = 'Reject'  # 排斥

        # 将表情转化为对应的文件夹
        if output == 'Reject':
            label = 0
        elif output == 'Despise':
            label = 1
        elif output == 'Listen':
            label = 2
        elif output == 'Understand':
            label = 3
        elif output == 'Puzzled':
            label = 4
        elif output == 'Happy':
            label = 5
        else:
            label = 6  # output为Else

        # 构造目标文件夹路径
        target_folder = str(label)

        # 移动图像文件到相应的文件夹中
        os.rename(image_path, os.path.join(target_folder, image_file))
