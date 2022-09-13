import cv2
import sys

src = cv2.imread('rose.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('영상을 불러올 수 없습니다')
    sys.exit()

dst = cv2.GaussianBlur(src, (0, 0), 3)
dst2 = cv2.blur(src, (7, 7))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.waitKey()

cv2.destroyAllWindows()

