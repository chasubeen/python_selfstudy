### 1. 라이브러리 import

import io
import json
import requests # Flask 서버에 POST 요청 생성
from PIL import Image # 샘플 입력 이미지 파일 읽어들이기

from torchvision import transforms # 입력 이미지 배열 전처리

### 2. 이미지 파일 읽어오기

image = Image.open("./digit_image.jpg")

### 3. 전처리

## 전처리 함수 정의
# 이미지를 모델이 읽을 수 있는 포맷으로 변환
def image_to_tensor(image):
    gray_image = transforms.functional.to_grayscale(image) # RGB -> 흑백
    resized_image = transforms.functional.resize(gray_image, (28, 28)) # 이미지 사이즈 조절
    input_image_tensor = transforms.functional.to_tensor(resized_image) # torch.tensor로 캐스팅
    # 이미지 정규화
    input_image_tensor_norm = transforms.functional.normalize(input_image_tensor, (0.1302,), (0.3069,))
    
    return input_image_tensor_norm

image_tensor = image_to_tensor(image) # 플라스크 서버에 입력 데이터로 전송되는 내용

### 4. 데이터 전송
# 수신측 플라스크 서버가 픽셀 값의 스트림을 이미지로 재구성하는 방법을 알 수 있도록
# 이미지 pixel 값과 이미지 형태를 모두 전송

# 데이터 직렬화
dimensions = io.StringIO(json.dumps({'dims': list(image_tensor.shape)}))
data = io.BytesIO(bytearray(image_tensor.numpy()))

### 5. 요청 생성
# r: 플라스크 서버에서 보내는 요청에 대한 응답 수신
# 이는 후처리된 모델 예측을 포함
r = requests.post('http://localhost:8890/test',
                  files={'metadata': dimensions, 'data' : data})

### 6. 출력 읽어들이기
response = json.loads(r.content) # Flask 서버의 출력(0 ~ 9)을 저장

### 7. 응답 출력
print("Predicted digit :", response)



























