import cv2
import sys

cap1 = cv2.VideoCapture('woman.mp4')
cap2 = cv2.VideoCapture('raining.mp4')

if not cap1.isOpened() or not cap2.isOpened():
    print('동영상을 불러올 수 없습니다')
    sys.exit()

frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
print('frame_cnt1:', frame_cnt1)
print('frame_cnt2:', frame_cnt2)

fps = cap1.get(cv2.CAP_PROP_FPS)
print('fps: ', fps)
delay = int(1000 / fps)
print('delay', delay)

while True:
    ret1, frame1 = cap1.read()

    if not ret1:
        break

    ret2, frame2 = cap2.read()

    if not ret2:
        break

    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (50, 150, 0), (70, 255, 255))
    cv2.copyTo(frame2, mask, frame1)
    cv2.imshow('frame', frame1)
    key = cv2.waitKey(delay)

    if key == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
