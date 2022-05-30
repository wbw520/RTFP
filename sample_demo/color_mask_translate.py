import cv2
from utils.base_tools import get_name
from utils.translation_tools import read_json, PolygonTrans
import os
from PIL import Image
import shutil


def main():
    root = "/home/wangbowen/DATA/Facade/"
    item_list = get_name(root, mode_folder=False)
    image_list = []
    for item in item_list:
        name = item.split(".")[0]
        if name not in image_list:
            image_list.append(name)

    for img in image_list:
        print(img)
        json_root = root + "/" + img + ".json"
        if not os.path.exists(json_root):
            print("file not exist: ", json_root)
            continue

        out = read_json(json_root)
        if "IMG_E" in img:
            suffix = ".JPG"
        else:
            suffix = ".jpg"

        image = cv2.imread(root + "/" + img + suffix, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape

        PT = PolygonTrans()
        mask = PT.polygon2mask((h, w), out)
        color_map = PT.id2trainId(mask)

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        color_map = Image.fromarray(color_map)

        image.save(root_img + "/" + img + ".png")
        mask.save(root_binary_mask + "/" + img + ".png")
        color_map.save(root_color_mask + "/" + img + ".png")


if __name__ == '__main__':
    p_root = "/home/wangbowen/DATA/facades/"
    use_predict = ""
    shutil.rmtree(p_root + use_predict + "translated_data", ignore_errors=True)
    root_img = p_root + use_predict + "translated_data/images"
    root_color_mask = p_root + use_predict + "translated_data/color_mask"
    root_binary_mask = p_root + use_predict + "translated_data/binary_mask"
    os.makedirs(root_img, exist_ok=True)
    os.makedirs(root_color_mask, exist_ok=True)
    os.makedirs(root_binary_mask, exist_ok=True)
    main()