import cv2
import sys
import numpy as np

src = cv2.imread('namecard.jpg')

if src is None:
    print('영상을 불러올 수 없습니다')
    sys.exit()

w, h = 720, 400

srcQuad = np.array([[345, 287], [1123, 229], [1272, 657], [332, 768]], np.float32)
dstQuad = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)

dst = cv2.warpPerspective(src, pers, (w, h))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()



