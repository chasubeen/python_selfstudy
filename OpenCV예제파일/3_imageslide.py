import cv2
import sys
import glob

imgs = glob.glob('images\\*.jpg')
# print(imgs)

if not imgs:
    print('영상을 불러올 수 없습니다')
    sys.exit()

idx = 0

while True:
    img = cv2.imread(imgs[idx])

    if img is None:
        print('영상을 불러올 수 없습니다')
        break

    cv2.imshow('image', img)
    if cv2.waitKey(1000) >= 0:
        break

    idx += 1
    if idx >= len(imgs):
        idx = 0

cv2.destroyAllWindows()