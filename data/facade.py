import torch
from PIL import Image
import numpy as np
from utils.base_tools import get_name
import json
from sklearn.model_selection import train_test_split
import cv2


ignore_label = 255
num_classes = 10
colors = {0: [0, 0, 0], 1: [70, 70, 70], 2: [250, 170, 30], 3: [70, 130, 180], 4: [0, 60, 100], 5: [153, 153, 153],
          6: [107, 142, 35], 7: [255, 0, 0], 8: [0, 0, 142], 9: [220, 220, 0]}


class PolygonTrans():
    def __init__(self):
        self.binary = {"building": 1, "window": 2, "sky": 3, "roof": 4, "door": 5, "tree": 6, "people": 7, "car": 8, "sign": 9}
        self.overlap_order = ["sky", "building", "roof", "door", "window", "tree", "people", "car", "sign"]

    def polygon2mask(self, img_size, polygons, rectangles):
        mask = np.zeros(img_size, dtype=np.uint8)
        for cat in self.overlap_order:
            polygon = polygons[cat]
            cv2.fillPoly(mask, polygon, color=self.binary[cat])
            rectangle = rectangles[cat]
            for ret in rectangle:
                x1, y1 = ret[0]
                x2, y2 = ret[1]
                mask[y1:y2, x1:x2] = self.binary[cat]
        return mask

    # translate label_id to color img
    def id2trainId(self, label):
        w, h = label.shape
        label_copy = np.zeros((w, h, 3), dtype=np.uint8)
        for index, color in colors.items():
            label_copy[label == index] = color
        return label_copy.astype(np.uint8)


def read_json(file_name):
    record = {"building": [], "window": [], "sky": [], "roof": [], "door": [], "tree": [], "people": [], "car": [], "sign": []}
    record_rectangle = {"building": [], "window": [], "sky": [], "roof": [], "door": [], "tree": [], "people": [], "car": [], "sign": []}
    with open(file_name, "r") as load_polygon:
        data = json.load(load_polygon)

    data = data["shapes"]
    for item in data:
        label = item["label"]
        points = item["points"]
        shape = item["shape_type"]
        if label not in record:
            continue

        if shape == "rectangle":
            record_rectangle[label].append(np.array(points, dtype=np.int32))
        else:
            record[label].append(np.array(points, dtype=np.int32))
    return record, record_rectangle


def prepare_facade_data(args):
    roots = args.root + "Facade/translated_data/"
    items = get_name(roots + "images", mode_folder=False)
    record = []
    for item in items:
        record.append([roots + "images/" + item, roots + "binary_mask/" + item])

    train, other = train_test_split(record, train_size=0.8, random_state=1)
    val, test = train_test_split(record, train_size=0.5, random_state=1)
    return {"train": train, "val": val, "test": test}


class Facade(torch.utils.data.Dataset):
    def __init__(self, args, mode, joint_transform=None, standard_transform=None):
        self.args = args
        self.imgs = prepare_facade_data(args)[mode]
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.joint_transform = joint_transform
        self.standard_transform = standard_transform
        if self.args.use_ignore:
            self.id_to_trainid = {6: 255, 7: 255, 8: 255, 9: 255}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        if self.args.use_ignore:
            for k, v in self.id_to_trainid.items():
                mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.standard_transform is not None:
            img = self.standard_transform(img)

        return {"images": img, "masks": torch.from_numpy(np.array(mask, dtype=np.int32)).long()}

    def __len__(self):
        return len(self.imgs)