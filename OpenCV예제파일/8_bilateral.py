import cv2
import sys

src = cv2.imread('lenna.bmp')

if src is None:
    print('영상을 불러올 수 없습니다')
    sys.exit()

dst = cv2.bilateralFilter(src, -1, 10, 5)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()

