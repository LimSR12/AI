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
images, labels = next(dataiter)

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
