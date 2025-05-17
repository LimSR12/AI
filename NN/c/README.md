# NN 구현 - C
## Neural Net을 Python 라이브러리 없이 C로 구현
```
main
 └─ 배열 선언 및 초기화
 └─ 학습률 및 인식률 기록용 변수 설정
 └─ trainNeuralNet()
     ├─ forwardPropagation()
     └─ backPropagation()
 └─ saveMetaData()
 └─ 메모리 해제
```
![image](https://github.com/user-attachments/assets/6bb59c17-607e-44cb-b9c0-fc524c59b36d)

![image](https://github.com/user-attachments/assets/c2310183-3c02-44ab-a7dc-bf8434edd09c)

---
## 인식률 개선 로직 추가
```
main
 └─ 배열 선언 및 초기화
 └─ 가중치(weight) 및 모멘텀(velocity) 배열 선언 및 초기화
 └─ 학습률(learning_rate) 및 인식률(epoch_accuracy) 변수 설정
 └─ for문 (epoch 반복)
     ├─ trainNeuralNet()
         ├─ 파일 하나씩 읽고 input 배열에 저장
         ├─ forwardPropagation()
             ├─ input → hidden1 → hidden2 → hidden3 → output 계산
             ├─ 출력층에 Softmax 함수 적용
         ├─ target 벡터 설정
         ├─ backPropagation()
             ├─ 출력층 오차 계산 (Cross Entropy 기반)
             ├─ hidden layer 오차 전파
             ├─ learning_rate 및 모멘텀(Momentum) 적용해 weight 업데이트
     ├─ testAccuracy()
         ├─ 테스트 데이터로 forwardPropagation만 수행
         ├─ 예측값과 실제값 비교하여 인식률 계산
     ├─ Early Stopping 판단
         ├─ 인식률이 best_accuracy를 갱신하면 patience_counter 초기화
         ├─ 인식률이 개선되지 않으면 patience_counter 증가
         ├─ patience_limit 초과 시 학습 조기 종료
     ├─ best model 저장 (saveMetaData)
         ├─ 가장 좋은 인식률을 기록한 가중치 및 설정값 저장
 └─ 메모리 해제 (가중치, 모멘텀 배열 free)
```

![image](https://github.com/user-attachments/assets/2e348e04-9023-4a0a-9e11-089abcf1d0b2)


![image](https://github.com/user-attachments/assets/32fe13f1-ec9e-40b9-9924-b0f3114ae939)
