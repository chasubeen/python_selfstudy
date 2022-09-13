import cv2
import sys
import matplotlib.pyplot as plt

src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('영상을 읽어올 수 없습니다')
    sys.exit()

hist = cv2.calcHist([src], [0], None, [256], [0, 256])

cv2.imshow('src', src)
plt.plot(hist)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

