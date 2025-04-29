# NN 구현 - C / Python

1. 프로젝트 소개

C언어로 구현한 인공 신경망(Neural Network) 기반 MNIST 필기체 숫자 인식 프로그램입니다.  

Raw 이미지 파일(28x28 픽셀)을 입력으로 받아, 다층 퍼셉트론(Multi-Layer Perceptron, MLP) 구조를 통해 0~9 숫자를 분류합니다.

2. 프로젝트 디렉토리 구조
```
## Directory Structure
├── mnist_raw
│   ├── training
│   │   └── 0~9 (training data folders)
│   └── testing
│       └── 0~9 (testing data folders)
├── neural_net_model
│   └── [ 신경망 학습습 모델 ]
├── neural_net_testing_model
│   └── [ 테스트 결과 모델 ]
├── trainmodel.c
├── testmodel.c
├── meta_h1.txt / meta_h2.txt / meta_h3.txt
```

3. 기능 소개
```
- `trainmodel.c`: MNIST training data를 사용해 신경망 학습 및 가중치 저장
- `testmodel.c`: 학습된 메타데이터(meta_h*.txt)를 불러와 테스트 데이터 인식 및 정확도 출력
- `meta_h1.txt`, `meta_h2.txt`, `meta_h3.txt`: 은닉층 개수별 저장된 학습 가중치 파일
```

4. 신경망 구조
```
- 입력층(Input Layer): 784 nodes (28x28 이미지 픽셀)
- 은닉층(Hidden Layers): 선택 가능
  - 1 hidden layer: 256 nodes
  - 2 hidden layers: 256 → 128 nodes
  - 3 hidden layers: 256 → 128 → 64 nodes
- 출력층(Output Layer): 10 nodes (숫자 0~9)
- 활성화 함수: Sigmoid, Softmax
- 손실 함수: Cross Entropy 기반 (or MSE 기반)
- 최적화 방법: 경사하강법 (Gradient Descent) + 오류역전파(Backpropagation)
```

5. 가중치 초기화 및 학습 설정
```
- 가중치 초기값: -0.5 ~ +0.5 범위에서 랜덤 초기화
- 학습 Epoch 수: 40 epochs
- 배치 크기(Batch Size): 1 (Online Learning)
- 학습률(Learning Rate): 고정값 사용
```