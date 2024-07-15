### 0. 라이브러리 import

import os
import json
import numpy as np

from flask import Flask, request
import torch
import torch.nn as nn
import torch.nn.functional as F

### 1. 모델 클래스(아키텍처) 정의

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = nn.Conv2d(1, 16, 3, 1)
        self.cn2 = nn.Conv2d(16, 32, 3, 1)
        self.dp1 = nn.Dropout2d(0.10)
        self.dp2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 64) # 4608 is basically 12 X 12 X 32
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.cn1(x)
        x = F.relu(x)
        
        x = self.cn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dp1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)

        x = self.fc2(x)

        op = F.log_softmax(x, dim=1)

        return op

### 2. 모델 객체 생성

# 사전 훈련된 모델 매개변수 로딩
model = ConvNet()

PATH_TO_MODEL = "./convnet.pth"
model.load_state_dict(torch.load(PATH_TO_MODEL, map_location="cpu"))
model.eval() # 평가 모드로 설정 -> 모델 매개변수 튜닝 x

### 3. 추론 파이프라인 구성

# 예측을 위한 함수
def run_model(input_tensor):
    model_input = input_tensor.unsqueeze(0)
    with torch.no_grad():
        model_output = model(model_input)[0]
    model_prediction = model_output.detach().numpy().argmax()
    return model_prediction

# 예측 결과(정수)를 문자열로 변환하는 함수
def post_process(output):
    return str(output)

### 4. 플라스크 앱 구축

# 플라스크 앱 인스턴스화
app = Flask(__name__)

# 서버 endpoint 기능 정의
@app.route("/test", methods=["POST"])
def test():
    ## POST 요청을 받아 어떠한 작업을 처리할 지 정의
    data = request.files['data'].read()
    md = json.load(request.files['metadata'])
    input_array = np.frombuffer(data, dtype=np.float32)
    input_image_tensor = torch.from_numpy(input_array).view(md["dims"])

    output = run_model(input_image_tensor)
    final_output = post_process(output)

    return final_output

# 서버 hosting(Serving)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8890)











