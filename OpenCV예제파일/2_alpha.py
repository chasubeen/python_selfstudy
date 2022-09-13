import cv2
import sys

src = cv2.imread('field.bmp')
hero = cv2.imread('hero.png', cv2.IMREAD_UNCHANGED) # png 영상

if src is None or hero is None:
    print('영상을 불러올 수 없습니다')
    sys.exit()

mask = hero[:, :, 3]
hero = hero[:, :, :-1] # hero는 b, g, r 3채널, png는 4채널 (b, g, r, a)
h, w = mask.shape[:2]
crop = src[10:10+h, 10:10+w]

cv2.copyTo(hero, mask, crop)

cv2.imshow('src', src)
cv2.imshow('mask', mask)
cv2.imshow('hero', hero)
cv2.waitKey()
cv2.destroyAllWindows()

