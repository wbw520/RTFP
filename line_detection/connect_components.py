import cv2
import numpy as np
from inference import show_single


img = cv2.imread("../demo/test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
bin_clo = cv2.erode(binary, kernel2, iterations=1)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)

# print('num_labels = ', num_labels)
# print('labels = ', labels)
# # 不同的连通域赋予不同的颜色
output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

for i in range(1, num_labels):
    mask = labels == i
    # 连通域的信息：对应各个轮廓的x、y、width、height和面积
    print('stats = ', stats[i])
    if stats[i][4] < 30:
        continue
    # 连通域的中心点
    print('centroids = ', centroids[i])
    output[:, :, 0][mask] = np.random.randint(0, 255)
    output[:, :, 1][mask] = np.random.randint(0, 255)
    output[:, :, 2][mask] = np.random.randint(0, 255)
    break

show_single(output)
