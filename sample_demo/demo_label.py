import cv2
from data.facade import read_json, PolygonTrans
from utils.demo_tools import show_single


file_name = "42885583962_e6ee1f7de1_k"
root = "/home/wangbowen/DATA/Facade/" + file_name + ".json"
out, out2 = read_json(root)
print(len(out2["window"]))
image = cv2.imread("/home/wangbowen/DATA/Facade/" + file_name + ".jpg", cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, c = image.shape

PT = PolygonTrans()
mask = PT.polygon2mask((h, w), out, out2)
color_map = PT.id2trainId(mask)
show_single(color_map, save=False)