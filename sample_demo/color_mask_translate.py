import cv2
from utils.base_tools import get_name
from data.facade import read_json, PolygonTrans
import os
from PIL import Image
import shutil


def main():
    item_list = get_name(root + name, mode_folder=False)
    image_list = []
    for item in item_list:
        name_ = item.split(".")[0]
        if name_ not in image_list:
            image_list.append(name_)

    for img in image_list:
        print(img)
        json_root = root + name + "/" + img + ".json"
        if not os.path.exists(json_root):
            print("file not exist: ", json_root)
            continue

        polygons, rectangles = read_json(json_root)
        if "IMG_E" in img:
            suffix = ".JPG"
        else:
            suffix = ".jpg"

        image = cv2.imread(root + name + "/" + img + suffix, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape

        PT = PolygonTrans()
        mask = PT.polygon2mask((h, w), polygons, rectangles)
        color_map = PT.id2trainId(mask)

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        color_map = Image.fromarray(color_map)

        image.save(root_img + "/" + img + ".jpg")
        mask.save(root_binary_mask + "/" + img + ".jpg")
        color_map.save(root_color_mask + "/" + img + ".jpg")


if __name__ == '__main__':
    root = "/home/wangbowen/DATA/Facade/"
    name = "wang"
    use_predict = ""
    shutil.rmtree(root + name + "translated_data", ignore_errors=True)
    root_img = root + name + "_translated_data/images"
    root_color_mask = root + name + "_translated_data/color_mask"
    root_binary_mask = root + name + "_translated_data/binary_mask"
    os.makedirs(root_img, exist_ok=True)
    os.makedirs(root_color_mask, exist_ok=True)
    os.makedirs(root_binary_mask, exist_ok=True)
    main()