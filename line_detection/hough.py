import cv2
import numpy as np
from inference import show_single

src = cv2.imread("/home/wangbowen/DATA/Facade/translated_data/images/IMG_E1283.png")
# src = cv2.imread("../demo/test.png")
gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


dst = cv2.equalizeHist(gray_img)
# 高斯滤波降噪
gaussian = cv2.GaussianBlur(dst, (9, 9), 0)
# cv.imshow("gaussian", gaussian)

# 边缘检测
edges = cv2.Canny(gaussian, 70, 150)
show_single(edges)

# Hough 直线检测
# 重点注意第四个参数 阈值，只有累加后的值高于阈值时才被认为是一条直线，也可以把它看成能检测到的直线的最短长度（以像素点为单位）
# 在霍夫空间理解为：至少有多少条正弦曲线交于一点才被认为是直线
lines = cv2.HoughLinesP(edges, 1, 1 * np.pi / 180, 10, minLineLength=10, maxLineGap=5)#统计概率霍夫线变换函数：图像矩阵，极坐标两个参数，一条直线所需最少的曲线交点，组成一条直线的最少点的数量，被认为在一条直线上的亮点的最大距离
print("Line Num : ", len(lines))

# 画出检测的线段
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(src, (x1, y1), (x2, y2), (255, 0, 0), 2)
    pass

show_single(src)