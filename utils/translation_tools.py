import cv2
import numpy as np
import json


colors = {0: [0, 0, 0], 1: [70, 70, 70], 2: [250, 170, 30], 3: [70, 130, 180], 4: [0, 60, 100], 5: [153, 153, 153],
          6: [107, 142, 35], 7: [255, 0, 0], 8: [0, 0, 142], 9: [220, 220, 0]}


class PolygonTrans():
    def __init__(self):
        self.binary = {"building": 1, "window": 2, "sky": 3, "roof": 4, "door": 5, "tree": 6, "people": 7, "car": 8, "sign": 9}
        self.overlap_order = ["building", "window", "sky", "roof", "door", "tree", "people", "car", "sign"]

    def polygon2mask(self, img_size, polygons):
        mask = np.zeros(img_size, dtype=np.uint8)
        for cat in self.overlap_order:
            polygon = polygons[cat]
            cv2.fillPoly(mask, polygon, color=self.binary[cat])
        return mask

    # trainslate label_id to color img
    def id2trainId(self, label):
        w, h = label.shape
        label_copy = np.zeros((w, h, 3), dtype=np.uint8)
        for index, color in colors.items():
            label_copy[label == index] = color
        return label_copy.astype(np.uint8)


def read_json(file_name):
    record = {"building": [], "window": [], "sky": [], "roof": [], "door": [], "tree": [], "people": [], "car": [], "sign": []}
    with open(file_name, "r") as load_polygon:
        data = json.load(load_polygon)

    data = data["shapes"]
    for item in data:
        label = item["label"]
        points = item["points"]
        if label not in record:
            continue
        record[label].append(np.array(points, dtype=np.int32))
    return record