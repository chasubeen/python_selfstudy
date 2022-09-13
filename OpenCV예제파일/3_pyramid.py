import cv2
import sys

src = cv2.imread('cat.bmp')

if src is None:
    print('영상을 불러올 수 없습니다')
    sys.exit()

# rectangle tuple
rc = (250, 120, 200, 200)

cpy = src.copy()
cv2.rectangle(cpy, rc, (0, 0, 255), 2)
cv2.imshow('src', cpy)
cv2.waitKey()

for i in range(1, 4):
    src = cv2.pyrDown(src)
    cpy = src.copy()
    cv2.rectangle(cpy, rc, (0, 0, 255), 2, shift=i)
    cv2.imshow('src', cpy)
    cv2.waitKey()
    cv2.destroyWindow('src')

cv2.destroyAllWindows()



