import os
import numpy as np
import cv2
from tqdm import tqdm

# 创建保存图像的文件夹
os.makedirs("image", exist_ok=True)

# 读取CSV文件
csv_file = "fer2013.csv"

# 读取CSV中的像素值
with open(csv_file, 'r') as f:
    lines = f.readlines()

# 第一行是标头，跳过它
lines = lines[1:]

# 获取总行数
total_lines = len(lines)

# 遍历每一行，解析像素值并转换为图像
for i, line in enumerate(tqdm(lines)):
    # 按逗号分隔每个像素值
    pixel_values = line.strip().split(',')

    # 获取标签和像素值
    label = int(pixel_values[0])
    pixels = np.array(pixel_values[1].split(), dtype='uint8')

    # 将像素值转换为48x48的灰度图像
    image = pixels.reshape((48, 48))

    # 构造图像文件保存路径
    filename = os.path.join("image", f"image_{i}.png")

    # 将图像保存到文件中
    cv2.imwrite(filename, image)
