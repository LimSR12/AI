## 🧠 MNIST 손글씨 분류기 (NN & CNN)
이 프로젝트는 PyTorch를 기반으로 MNIST 손글씨 데이터셋을 분류하는 두 가지 모델을 구현한 예제입니다:

기본 신경망 모델 (Fully Connected Neural Network, nn_model.py)

합성곱 신경망 모델 (Convolutional Neural Network, cnn.py)

데이터셋 시각화 도구 (nn_setup.py)

<br/>
📁 프로젝트 구성

```📦 MNIST_Classification
├─ nn_setup.py         # MNIST 이미지 시각화 및 전처리
├─ nn_model.py         # 3층 Fully Connected Neural Network 모델 정의 및 학습
├─ cnn.py              # CNN 기반 모델 정의, 학습, 평가 및 모델 저장
└─ README.md
```
## 🧪 1. Fully Connected Neural Network (nn_model.py)
📌 구조
- 입력층: 28 × 28 → 784차원

- 은닉층 1: 512 노드 (ReLU)

- 은닉층 2: 512 노드 (ReLU)

- 출력층: 10개 클래스 (0~9 숫자)

⚙️ 학습 설정
- Optimizer: Adam

- Loss: CrossEntropyLoss

- Epochs: 5

- Batch size: 32

🏁 실행 방법

```
python nn_model.py
```
🎯 결과 예시
```
Epoch 1, Batch 200, Loss: 0.340
...
Accuracy of the network on the 10000 test images: 97.1%
```

## 🧠 2. Convolutional Neural Network (cnn.py)
## 📌 구조
- Conv1: 1채널 입력 → 32채널 (3×3 kernel, padding='same') + MaxPool

- Conv2: 32채널 → 64채널 (3×3 kernel, padding='same') + MaxPool

- Dropout(0.25) 후 Flatten

- FC1: 3136 → 1000

- FC2: 1000 → 10

- Activation: ReLU + 마지막 출력은 log_softmax

## ⚙️ 학습 설정
- Optimizer: Adam

- Loss: CrossEntropyLoss

- Epochs: 15

- Batch size: 32

- Device: GPU 지원

## 💾 모델 저장
모델 학습 후 Google Drive에 .pt 형식으로 저장 (CNN.pt)

## 🏁 실행 방법
```
python cnn.py
```
🎯 결과 예시
```
========= epoch[0] =========
Epoch 0 마지막 배치 Loss: 0.023
...
Test set Accuracy : 99.2%
```
## 🖼 3. 데이터 시각화 (nn_setup.py)
학습용 이미지 일부를 grid 형태로 시각화하여 확인할 수 있는 코드입니다.

실행 시 4장의 MNIST 이미지와 라벨이 출력됩니다.

🏁 실행 방법
```
python nn_setup.py
```