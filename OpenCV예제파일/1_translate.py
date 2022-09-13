import cv2
import sys
import numpy as np

src = cv2.imread('tekapo.bmp')

if src is None:
    print('영상을 불러올 수 없습니다')
    sys.exit()

aff = np.array([[1, 0, 200], [0, 1, 100]], dtype=np.float32)

dst = cv2.warpAffine(src, aff, (0, 0))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

