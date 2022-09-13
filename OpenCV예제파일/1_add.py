import cv2
import sys

src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('영상을 불러올 수 없습니다')
    sys.exit()

dst = cv2.add(src, 100)

src2 = cv2.imread('lenna.bmp')

if src2 is None:
    print('영상을 불러올 수 없습니다')
    sys.exit()

dst2 = cv2.add(src2, (100, 100, 100, 0))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('src2', src2)
cv2.imshow('dst2', dst2)
cv2.waitKey()




