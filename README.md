# 세종대학교 K-MOOC - 생성형 인공지능 입문 (김용국)
>## 6주차 과제
  >[과제 내용]
>
  >다층구조 신경망(MLP)를 이용해 MNIST 데이터세트 중
>
  >0 부터 9까지 필기숫자를 인식하는 코드를 작성하고 결과물을 제출하시오.

>## 소스코드
```python
# 모듈 import 및 데이터 load
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transform
```

```python
# 신경망 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
         # 첫 번째 완전 연결층 (입력: 28*28 픽셀, 출력: 512)
        self.fc1 = nn.Linear(28 * 28, 512)
        # 두 번째 완전 연결층 (입력: 512, 출력: 512)
        self.fc2 = nn.Linear(512, 512)
        # 세 번째 완전 연결층 (입력: 512, 출력: 10) - 10개의 숫자를 예측
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        # 입력 데이터를 1차원으로 펼침
        x = x.view(-1, 28 * 28)
        # 첫 번째 완전 연결층과 ReLU 활성화 함수
        x = torch.relu(self.fc1(x))
        # 두 번째 완전 연결층과 ReLU 활성화 함수
        x = torch.relu(self.fc2(x))
        # 세 번째 완전 연결층
        x = self.fc3(x)
        return x
```

```python

# 모델 훈련 및 평가 함수
def train_and_evaluate():
	  # 데이터 전처리: 텐서 변환 및 정규화
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        
     # 훈련 데이터셋 로드
    trainset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
        )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0
        )
    # 테스트 데이터셋 로드
    testset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
        )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0)
    
    # 모델 초기화
    net = Net()
    # 손실 함수 정의 (교차 엔트로피)
    criterion = nn.CrossEntropyLoss()
    # 최적화 알고리즘 정의 (Adam)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # 훈련 루프 (5 에포크 동안)
    for epoch in range(5):
        running_loss = 0.0
        # 훈련 데이터셋 반복
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # 매 200 배치마다 손실 출력
            if i % 200 == 199:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')
```

```python
# 테스트 데이터셋 평가
    correct = 0
    total = 0
    with torch.no_grad():  # 평가 과정에서는 gradient 계산 비활성화
        for data in testloader:
            images, labels = data  # 테스트 데이터와 라벨
            outputs = net(images)  # 모델에 테스트 데이터 통과
            _, predicted = torch.max(outputs.data, 1)  # 예측 값 얻기
            total += labels.size(0)  # 총 라벨 수
            correct += (predicted == labels).sum().item()  # 맞게 예측한 수

    # 정확도 출력
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

# 메인 모듈 보호
if __name__ == "__main__":
    train_and_evaluate()
```

>## 이미지 데이터

`torchvision.datasets.MNISt`를 사용하여 데이터를 다운로드하고 로드하면, 데이터는 `root` 매개변수에 지정된 경로에 저장된다. 

소스코드에서 `root='./data'`로 지정했으므로, 

MNIST 데이터셋은 현재 작업 디렉토리의 `data` 폴더에 저장된다.

MNIST 데이터셋의 원본 이미지를 확인하면 아래와 같다.

```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 데이터 전처리: 텐서 변환 및 정규화
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 훈련 데이터셋 로드
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

# 일부 훈련 데이터를 가져옴
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 이미지 시각화를 위한 함수 정의
def imshow(img):
    img = img / 2 + 0.5  # 정규화 해제
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 배치 중 첫 번째 이미지를 출력
print(' '.join('%5s' % labels[j].item() for j in range(4)))
imshow(torchvision.utils.make_grid(images))

# 메인 모듈 보호
if __name__ == "__main__":
    # 위에 정의된 코드를 실행하여 원본 이미지를 출력
    imshow(torchvision.utils.make_grid(images))

```

<img src="Figure_1.png" alt="dataSet figure image" />

>## 결과 도출
## MNIST 데이터셋 인식 훈련 결과 도출 사항

1. **훈련 중 손실 출력**:

각 에포크(epoch) 동안의 손실 값이 주기적으로 출력된다.
    
    ```python
    Epoch 1, Batch 200, Loss: 0.675
    Epoch 1, Batch 400, Loss: 0.353
    Epoch 1, Batch 600, Loss: 0.291
    
    ...
    
    Epoch 5, Batch 1400, Loss: 0.085
    Epoch 5, Batch 1600, Loss: 0.073
    Epoch 5, Batch 1800, Loss: 0.085
    ```
    
    매 200 배치마다 손실 값을 출력하는 것으로, 손실 값이 점차 감소하는 것을 볼 수 있다.
    
2. **훈련 완료 메시지**:
    
    
    모든 에포크가 완료된 후, 다음 메시지가 출력된다:
    
    ```python
    Finished Training
    ```
    
3. **테스트 정확도 출력**:

  테스트 데이터셋에 대한 정확도가 출력된다.
    
    ```python
    Accuracy of the network on the 10000 test images: 97.36%.
    ```
