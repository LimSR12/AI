# 기본 라이브러리
import numpy as np
import matplotlib.pyplot as plt

# PyTorch 라이브러리 및 서브모듈
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# torchvision에서 MNIST 데이터셋 로딩 및 전처리
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 구글 드라이브 연동 (학습된 모델 저장 목적)
from google.colab import drive
# 구글 드라이브 마운트
drive.mount('/content/drive')

# parameter setting
batch_size = 32
learning_rate = 0.001
epoch = 15

# GPU setting
cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

print('Current cuda devie is', device)

# dataset 불러오기
train_data = datasets.MNIST( # 학습 데이터 6만개
    root = './data/train',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

test_data = datasets.MNIST( # 테스트 데이터 6천개
    root = './data/test',
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

train_loader = DataLoader(
    dataset = train_data,
    batch_size = batch_size,
    shuffle = True
)

test_loader = DataLoader(
    dataset = test_data,
    batch_size = batch_size,
    shuffle = False
)

# 딥러닝 모델 setting
class Convolution_Neural_Networks(nn.Module):
  def __init__(self):
    super(Convolution_Neural_Networks, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding='same')
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')

    self.dropout = nn.Dropout2d(0.25)
    self.relu = nn.ReLU()
    self.maxpooling = nn.MaxPool2d(kernel_size = 2)

    self.fc1 = nn.Linear(3136, 1000) # 7 * 7 * 64 = 3136
    self.fc2 = nn.Linear(1000, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxpooling(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxpooling(x)

    x = self.dropout(x)
    x = torch.flatten(x, 1)

    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)

    output = F.log_softmax(x, dim=1)
    return output

# 모델에 GPU 적용
model = Convolution_Neural_Networks().cuda()
# 모델에 CPU 적용
#model = Convolution_Neural_Networks().cpu()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Train
for epoch in range(epoch):
    print(f"========= epoch[{epoch}] =========")
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # 마지막 배치일 때만 출력
        if batch_idx == len(train_loader) - 1:
            print(f"Epoch {epoch} 마지막 배치 Loss: {loss.item()}")


# Test
model.eval()
correct = 0
with torch.no_grad():
  for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()
print("Test set Accuracy : {:.2f}%".format(100. * correct / len(test_loader.dataset)))

# 파일 경로 지정
PATH = '/content/drive/MyDrive/Colab Notebooks/CNN.pt'

# .pt 파일 로드
state_dict = torch.load(PATH)
torch.save(model.state_dict(), PATH)
# state_dict 출력 (모델의 가중치와 편향 값 확인 가능)
for param_tensor in state_dict:
  print(param_tensor, "\t", state_dict[param_tensor].size())

# 첫 번째 컨볼루션 레이어의 weight 값 출력
print("conv1 필터 계수:")
print(model.conv1.weight)  # 전체 32개의 필터, 각 필터는 1x3x3 크기

# 경로에 맞게 수정
state_dict = torch.load(PATH)

# 키 목록 확인
for key in state_dict.keys():
    print(key, state_dict[key].shape)