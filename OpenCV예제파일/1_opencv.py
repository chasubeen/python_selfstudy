import sys
import cv2

print('현재 opencv 버전 : ', cv2.__version__)

img = cv2.imread('cat.bmp')

if img is None:
    print('영상을 불러올 수 없습니다')
    sys.exit()

cv2.namedWindow('image') # 새 창의 이름을 image라고 설정
cv2.imshow('image', img) # image창에 img 영상을 출력
cv2.waitKey() # 키보드 입력을 받을 때까지 대기
cv2.destroyAllWindows() # 모든 창을 닫음



