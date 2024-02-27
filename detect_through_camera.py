# 导入库
import cv2
import dlib
import math
import numpy as np

# 打开摄像头
capture = cv2.VideoCapture(0)

# 获取人脸检测器
detector = dlib.get_frontal_face_detector()

# 获取人脸关键点检测器、
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    # 读取视频流
    ret, frame = capture.read()
    # 灰度转化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测
    faces = detector(gray, 1)
    # 绘制人脸的矩形框和关键点
    for face in faces:
        # 绘制矩形框
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        # 检测关键点
        shape = predictor(gray, face)
        # 获取关键点的坐标
        for pt in shape.parts():  # parts:获取68个点的坐标
            # pt：取出每个点的坐标
            pt_position = (pt.x, pt.y)
            # 绘制关键点
            cv2.circle(frame, pt_position, 2, (255, 0, 0), -1)

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

        # # 计算点19到点17和点21连线的向量
        # vec1 = np.array([x17 - x19, y17 - y19])
        # vec2 = np.array([x21 - x19, y21 - y19])
        # # 计算夹角
        # cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        # theta = np.arccos(cos_theta)
        # # 将弧度转换为角度c
        # c = math.degrees(theta)

        # # 计算17点与19点连线与水平线的夹角（弧度）
        # angle_17_19 = math.atan2(y17 - y19, x17 - x19)
        # # 计算21点与19点连线与水平线的夹角（弧度）
        # angle_21_19 = math.atan2(y21 - y19, x21 - x19)
        # # 计算点17和点21分别与点19连线的夹角c
        # angle_c = angle_21_19 - angle_17_19
        # # 将弧度转换为角度
        # c = math.degrees(angle_c)

        # 17、19和21点的坐标
        point_17 = np.array([x17, y17])
        point_19 = np.array([x19, y19])
        point_21 = np.array([x21, y21])
        # 计算向量
        vector_17_19 = point_17 - point_19
        vector_21_19 = point_21 - point_19
        # 计算向量的长度
        length_17_19 = np.linalg.norm(vector_17_19)
        length_21_19 = np.linalg.norm(vector_21_19)
        # 计算点乘
        dot_product = np.dot(vector_17_19, vector_21_19)
        # 计算夹角的余弦值
        cosine_angle = dot_product / (length_17_19 * length_21_19)
        # 计算夹角的弧度值
        angle_radians = np.arccos(cosine_angle)
        # 将弧度转换为角度
        c = np.degrees(angle_radians)

        # 计算脸的长度
        face_length = lowest_point - highest_point
        # 计算66点与62点的纵坐标差值占脸长的比例L
        L = abs(y66 - y62) / face_length

        # # 打印夹角
        # print(f"48点与66点连线与水平线的夹角a_left为：{a_left} 度")
        # print(f"54点与66点连线与水平线的夹角a_right为：{a_right} 度")
        # print(f"眼高与眼宽的比值b为：{b}")
        print(f"点17和点21分别与点19连线的夹角c为：{c} 度")
        # print(f"66点与62点的纵坐标差值占脸长的比例L为：{L}")

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
        print(output)
        # 表示表情类型
        cv2.putText(frame, output, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    if cv2.waitKey(1) == ord('q'):
        break

    # 显示效果
    cv2.imshow("face detection landmark", frame)

# 释放资源
capture.release()
cv2.destroyAllWindows()
