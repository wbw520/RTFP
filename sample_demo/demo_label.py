import cv2
from utils.translation_tools import read_json, PolygonTrans
from utils.demo_tools import show_single


file_name = "IMG_E1327"
root = "/home/wangbowen/DATA/facade/" + file_name + ".json"
out = read_json(root)
image = cv2.imread("/home/wangbowen/DATA/facade/" + file_name + ".JPG", cv2.IMREAD_COLOR)
print(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, c = image.shape

PT = PolygonTrans()
mask = PT.polygon2mask((h, w), out)
color_map = PT.id2trainId(mask)
show_single(color_map, save=False)