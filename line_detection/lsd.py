import cv2
from inference import show_single


src = cv2.imread("/home/wangbowen/DATA/Facade/translated_data/images/32052284_477d66a5ae_o.png")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
gray = cv2.GaussianBlur(gray, (5, 5), 5)
gray = cv2.GaussianBlur(gray, (3, 3), 5)

LSD = cv2.createLineSegmentDetector(0)
dlines = LSD.detect(gray)

for dline in dlines[0]:
    x0 = int(round(dline[0][0]))
    y0 = int(round(dline[0][1]))
    x1 = int(round(dline[0][2]))
    y1 = int(round(dline[0][3]))
    cv2.line(src, (x0, y0), (x1, y1), 255, 2, cv2.LINE_AA)

show_single(src, save=True, name="lsd.png")