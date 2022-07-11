import cv2
from data.facade import read_json, PolygonTrans
from utils.demo_tools import show_single
import numpy as np
from PIL import Image


file_name = "IMG_E1276"
root = "/home/wangbowen/DATA/Facade/raw_data/" + file_name + ".json"
out, out2 = read_json(root)
image = cv2.imread("/home/wangbowen/DATA/Facade/raw_data/" + file_name + ".JPG", cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, c = image.shape

PT = PolygonTrans()
mask = PT.polygon2mask((h, w), out, out2)
print(np.sum(mask == 10))
color_map = PT.id2trainId(mask)
mask = Image.fromarray(mask)
mask.save("tt.png")

img = Image.open("tt.png")
# img = img.resize((224, 112), Image.NEAREST)
img = np.array(img)
print(np.sum(img == 10))