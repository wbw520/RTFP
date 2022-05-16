import torch
from PIL import Image
import numpy as np
from utils.base_tools import get_name


ignore_label = 255


def prepare_facade_data(args):
    items = get_name(args.root + "/translated_data/images")


class Facade(torch.utils.data.Dataset):
    def __init__(self, args, mode, joint_transform=None, standard_transform=None):
        self.args = args
        self.imgs = ""
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