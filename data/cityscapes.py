import os
import numpy as np
import torch
from PIL import Image


num_classes = 19
longer_size = 2048
ignore_label = 255


def make_dataset_cityscapes(args, quality, mode):
    assert (quality == 'fine' and mode in ['train', 'val']) or \
           (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

    if quality == 'coarse':
        img_dir_name = 'leftImg8bit_trainextra' if mode == 'train_extra' else 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(args.root, 'gtCoarse', 'gtCoarse', mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        img_dir_name = 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(args.root, 'gtFine_trainvaltest', 'gtFine', mode)
        mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join(args.root, img_dir_name, 'leftImg8bit', mode)
    assert os.listdir(img_path) == os.listdir(mask_path)
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)
    return items


class CityScapes(torch.utils.data.Dataset):
    def __init__(self, args, quality, mode, joint_transform=None, standard_transform=None):
        self.imgs = make_dataset_cityscapes(args, quality, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.standard_transform = standard_transform
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
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


class ColorTransition(object):
    def __init__(self, ignore_label=19):
        self.color = {0: [128, 64, 128],   # class 0   road
                      1: [244, 35, 232],   # class 1   sidewalk
                      2: [70, 70, 70],     # class 2   building
                      3: [102, 102, 156],  # class 3   wall
                      4: [190, 153, 153],  # class 4   fence
                      5: [153, 153, 153],  # class 5   pole
                      6: [250, 170, 30],   # class 6   traffic light
                      7: [220, 220, 0],    # class 7   traffic sign
                      8: [107, 142, 35],   # class 8   vegetation
                      9: [152, 251, 152],  # class 9   terrain
                      10: [70, 130, 180],   # class 10  sky
                      11: [220, 20, 60],    # class 11  person
                      12: [255, 0, 0],      # class 12  rider
                      13: [0, 0, 142],      # class 13  car
                      14: [0, 0, 70],       # class 14  truck
                      15: [0, 60, 100],     # class 15  bus
                      16: [0, 80, 100],     # class 16  train
                      17: [0, 0, 230],      # class 17  motorcycle
                      18: [119, 11, 32]}    # class 18  bicycle

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                          3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                          7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                          14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                          18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                          28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def recover(self, image):  # convert predict of binary to color
        image = image.cpu().detach().numpy()
        color_image = self.id2trainId(image, reverse=True)
        return color_image.astype(np.uint8)

    # trainslate label_id to train_id for color img
    def id2trainId(self, label, reverse=False):
        if reverse:
            w, h = label.shape
            label_copy = np.zeros((w, h, 3), dtype=np.uint8)
            for index, color in self.color.items():
                label_copy[label == index] = color
        else:
            w, h, c = label.shape
            label_copy = np.zeros((w, h), dtype=np.uint8)
            for index, color in self.color.items():
                label_copy[np.logical_and(*list([label[:, :, i] == color[i] for i in range(3)]))] = index
        return label_copy

    # trainslate label_id to train_id for binary img
    def id2trainIdbinary(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy


