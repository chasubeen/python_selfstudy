### 라이브러리 import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import time
import argparse


### CNN 모델 아키텍처 정의
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
    
### 모델 훈련 루틴 정의    
def train(args):
    torch.manual_seed(0)

    device = torch.device("cpu") # 장치 설정

    # 데이터로더 정의
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../data', train = True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1302,), (0.3069,))
                        ])),
        batch_size=128, shuffle=True)  
    
    # 모델 객체 선언
    model = ConvNet()

    # 옵티마이저 설정
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    
    # 학습 mode
    model.train()

    ## 훈련 loop 수행
    for epoch in range(args.epochs):
        for b_i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred_prob = model(X)
            loss = F.nll_loss(pred_prob, y) # nll is the negative likelihood loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if b_i % 10 == 0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i, len(train_dataloader),
                    100. * b_i / len(train_dataloader), loss.item()))
         
## 실행 함수            
def main():
    # 파이썬 훈련 프로그램을 실행하는 동안 세대 수 등의 hyper-parameter 입력에 도움이 되는
    # 인수 Parser를 활용
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int)
    args = parser.parse_args()

    # 실행 시간 측정
    start = time.time()

    train(args)
    print(f"Finished training in {time.time()-start} secs")
    
# 해당 스크립트 실행 시 main()이 실행되는지 확인
if __name__ == '__main__':
    main()
    