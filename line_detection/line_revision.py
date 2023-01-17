import cv2
import numpy as np
import math
from inference import show_single
from shapely.geometry import LineString


def lsd():
    src = cv2.imread("/home/wangbowen/DATA/Facade/translated_data/images/32052284_477d66a5ae_o.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    gray = cv2.GaussianBlur(gray, (5, 5), 5)
    gray = cv2.GaussianBlur(gray, (3, 3), 5)

    LSD = cv2.createLineSegmentDetector(0)
    dlines = LSD.detect(gray)

    line_record = []

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        line_record.append([x0, y0, x1, y1])

    return line_record, src


def calc_abc_from_line_2d(x0, y0, x1, y1):
    a = y0-y1
    b = x1-x0
    c = x0*y1-x1*y0
    return a, b, c


def get_line_cross_point(line1, line2):
    a0, b0, c0 = calc_abc_from_line_2d(*line1)
    a1, b1, c1 = calc_abc_from_line_2d(*line2)
    D = a0*b1-a1*b0
    if D == 0:
        return None
    x = (b0*c1-b1*c0)/D
    y = (a1*c0-a0*c1)/D
    return x, y


def combine(lines):
    def get_line(j):
        if index[j] == 1:
            current_line = [lines[j][2], lines[j][3], lines[j][4], lines[j][5]]
        else:
            if j == 0 or j == 2:
                current_line = [5, lines[j][0], 20, lines[j][0]]
            else:
                current_line = [lines[j][0], 20, lines[j][0], 40]
        return current_line

    index = []
    for i in range(len(lines)):
        if len(lines[i]) == 1:
            index.append(0)
        else:
            index.append(1)

    if np.array(index).sum() < 2:
        return None

    cross_record = []
    start_line = None
    for s in range(len(index)):
        p_current_line = get_line(s)
        if s == 0:
            start_line = p_current_line

        if s == 3:
            p_next_line = start_line
        else:
            p_next_line = get_line(s+1)
        cross_point = get_line_cross_point(p_current_line, p_next_line)
        cross_record.append(cross_point)
    return cross_record


def line_search(enhance_stats, lines):
    x, y, w, h = enhance_stats
    top = LineString([(x, y), (x + w, y)])
    bottom = LineString([(x, y + h), (x + w, y + h)])
    left = LineString([(x, y), (x, y + h)])
    right = LineString([(x + w, y), (x + w, y + h)])

    line_list = {"top": top, "left": left, "bottom": bottom, "right": right}
    distance_thresh = 10
    degree_thresh = 0.1
    max_selection = 1
    record = {"top": [], "left": [], "bottom": [], "right": []}

    for key, value in line_list.items():
        # print(key)
        for (x0, y0, x1, y1) in lines:
            current_line = LineString([(x0, y0), (x1, y1)])
            current_degree = math.atan2(y0 - y1, x0 - x1)
            current_dis = value.distance(current_line)
            line_len = (x0 - x1)**2 + (y0 - y1)**2

            if current_dis > distance_thresh:
                continue

            # print([current_dis, current_degree, x0, y0, x1, y1])

            if key == "top" or key == "bottom":
                if math.pi * 1/5 < abs(current_degree) < math.pi * 4/5:
                    continue
                if line_len > w**2 * 1.5 or line_len < w**2 / 3:
                    continue
            else:
                if abs(current_degree) < math.pi * 1/3 or abs(current_degree) > math.pi * 2/3:
                    continue
                if line_len > h**2 * 1.5 or line_len < h**2 / 3:
                    continue

            status = True
            for i in range(len(record[key])):
                if abs(abs(abs(current_degree) - math.pi/2) - abs(abs(record[key][i][1]) - math.pi/2)) < degree_thresh:
                    if record[key][i][0] > current_dis:
                        record[key][i] = [current_dis, current_degree, x0, y0, x1, y1]
                    status = False

            if status:
                record[key].append([current_dis, current_degree, x0, y0, x1, y1])

    final_line = []
    for key2, value2 in record.items():
        value2.sort(key=lambda s: s[0], reverse=False)
        num = min(len(value2), max_selection)
        if num == 0:
            if key2 == "top":
                final_line.append([y])
            elif key2 == "bottom":
                final_line.append([y + h])
            elif key2 == "left":
                final_line.append([x])
            else:
                final_line.append([x + w])
            continue

        for j in range(num):
            final_line.append(value2[j])

    return final_line


def revision():
    img = cv2.imread("../demo/test.png")
    img = cv2.resize(img, (2048, 1152))
    img_orl = img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_clo = cv2.erode(binary, kernel1, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)

    # print('num_labels = ', num_labels)
    # print('labels = ', labels)
    # # 不同的连通域赋予不同的颜色
    lines, scr = lsd()

    for i in range(1, num_labels):
        # if i < 10:
        #     continue
        mask = labels == i
        # 连通域的信息：对应各个轮廓的x、y、width、height和面积
        if stats[i][4] < 100:
            continue
        # # 连通域的信息：对应各个轮廓的x、y、width、height和面积
        # print('stats = ', stats[i])
        # # 连通域的中心点
        # print('centroids = ', centroids[i])

        current_patch = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        current_patch[mask] = 255
        # show_single(current_patch)

        # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # current_patch = cv2.dilate(current_patch, kernel2, iterations=1)
        # show_single(current_patch)

        x, y, w, h = cv2.boundingRect(current_patch)
        # print(x, y, w, h)
        # cv2.rectangle(current_patch, (x, y), (x + w, y + h), (225, 0, 255), 2)
        # show_single(current_patch)

        detect_lines = line_search([x, y, w, h], lines)
        final_point = combine(detect_lines)

        if final_point is None:
            continue

        start_x, start_y = None, None
        for w in range(len(final_point)):
            x0, y0 = round(final_point[w][0]), round(final_point[w][1])
            if w == 0:
                start_x, start_y = x0, y0
            if w == 3:
                x1, y1 = start_x, start_y
            else:
                x1, y1 = round(final_point[w+1][0]), round(final_point[w+1][1])
            cv2.line(scr, (x0, y0), (x1, y1), 255, 2, cv2.LINE_AA)

    show_single(scr, save=True, name="lines_detect.png")


if __name__ == '__main__':
    revision()