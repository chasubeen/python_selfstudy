import cv2
import sys
import matplotlib.pyplot as plt

src = cv2.imread('lenna.bmp')

if src is None:
    print('영상을 읽어올 수 없습니다')
    sys.exit()

colors = ['b', 'g', 'r']
bgr_planes = cv2.split(src)

for (p, c) in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0, 256])
    plt.plot(hist, color=c)

cv2.imshow('src', src)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()


