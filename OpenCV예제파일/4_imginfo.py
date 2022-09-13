import cv2

img1 = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('cat.bmp', cv2.IMREAD_COLOR) # 기본값 컬러영상

print('img1의 type : ', type(img1)) # <class 'numpy.ndarray'>
print('img1의 shape : ', img1.shape) # (480, 640) (h, w)
print('img2의 shpae : ', img2.shape) # (480, 640, 3) (h, w, 3)
print('img2의 dtype : ', img2.dtype) # uint8

# (480, 640, 3)
h, w = img2.shape[:2] # h: 480, w: 640
print('img2의 사이즈 : {} * {}'.format(w, h))

if len(img1.shape) == 2:
    print('img1은 흑백 영상입니다')
elif len(img1.shape) == 3:
    print('img1은 컬러 영상입니다')

# for문으로 픽셀값을 변경하는 것은 매우 느림
# for y in range(h):
#     for x in range(w):
#         img1[y, x] = 30
#         img2[y, x] = (123, 77, 231) # BGR

img1[:,:] = 30
img2[:,:] = (123, 77, 231)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey()
cv2.destroyAllWindows()



