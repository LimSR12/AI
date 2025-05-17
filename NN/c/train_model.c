/*
directory architecture
├─mnist_raw
│  ├─testing
│  │  ├─0
│  │  ├─1
│  │  ├─2
│  │  ├─3
│  │  ├─4
│  │  ├─5
│  │  ├─6
│  │  ├─7
│  │  ├─8
│  │  └─9
│  └─training
│      ├─0
│      ├─1
│      ├─2
│      ├─3
│      ├─4
│      ├─5
│      ├─6
│      ├─7
│      ├─8
│      └─9
├─neural_net_model
│  ├─neural_net_model
│  │  └─x64
│  │      └─Debug
│  │          └─neural_net_model.tlog
│  └─x64
│      └─Debug
└─neural_net_testing_model
	├─neural_net_testing_model
	│  └─x64
	│      └─Debug
	│          └─neural_n.633f50e4.tlog
	└─x64
		└─Debug
*/
#pragma warning(disable:4996)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define INPUT_NODES (IMG_WIDTH * IMG_HEIGHT) // 784
#define OUTPUT_NODES 10

#define NUM_HIDDEN_LAYERS 2
#define HIDDEN1_NODES 256
#define HIDDEN2_NODES 128
#define HIDDEN3_NODES 64

#define WEIGHT1 (INPUT_NODES * HIDDEN1_NODES)
#define WEIGHT2 (HIDDEN1_NODES * HIDDEN2_NODES)
#define WEIGHT3 (HIDDEN2_NODES * HIDDEN3_NODES)
#define WEIGHT4 (HIDDEN3_NODES * OUTPUT_NODES)

// 가중치 초기값 범위
#define RAND_WEIGHT_MIN -0.5f
#define RAND_WEIGHT_MAX 0.5f

// 경로 설정
#define TRAIN_DIR "../../mnist_raw/training/"
#define ES_TEST_DIR "../../mnist_raw/testing/"
#define META_DIR "../../meta"
#define META1_DIR "../../meta_h1.txt"
#define META2_DIR "../../meta_h2.txt"
#define META3_DIR "../../meta_h3.txt"

#define EPOCH 40

// 모멘텀
#define MOMENTUM_ALPHA 0.95f
#define LEARNING_RATE 0.01f
#define HALF_LEARNING_RATE (LEARNING_RATE * 0.5f)

// +================================================ 함수 선언부 ================================================+
// initialize 함수
void initialize_weights(float* weight, int left, int right);
float* makeArray(int height, int width);

// 학습 메인 함수
void trainNeuralNet(
	float* input,
	float* hidden1, float* hidden2, float* hidden3,
	float* output, float* target,
	float* weight1, float* weight2, float* weight3, float* weight4,
	int num_of_hidden_layer, int epoch,
	// 모멘텀 -> velocity 배열
	float* velocity1, float* velocity2, float* velocity3, float* velocity4
);
void forwardPropagation(
	float* input,
	float* hidden1, float* hidden2, float* hidden3,
	float* output,
	float* weight1, float* weight2, float* weight3, float* weight4,
	int num_of_hidden_layer
);
void backPropagation(
	float* input, float* hidden1, float* hidden2, float* hidden3,
	float* output, float* target,
	float* weight1, float* weight2, float* weight3, float* weight4,
	int num_of_hidden_layer
);

// 시그모이드 함수
float sigmoid(float x);
float sigmoid_derivative(float x);

// 메타데이터 저장 관련 함수
void saveMetaData(
	float* hidden1, float* hidden2, float* hidden3,
	float* output, float* weight1, float* weight2, float* weight3, float* weight4,
	int img_width, int img_height, int output_nodes, int num_hidden_layer,
	int hidden1_nodes, int hidden2_nodes, int hidden3_nodes,
	float rand_weight_min, float rand_weight_max
);
void write_array(FILE* fp, const char* name, float* arr, int size);

// softmax 함수
void softmax(float* input, int length);

// Early Stopping 사용될 테스트 함수
float test_accuracy(
	const char* test_dir,
	int img_width, int img_height, int output_nodes, int num_hidden_layer,
	int hidden1_nodes, int hidden2_nodes, int hidden3_nodes,
	float* weight1, float* weight2, float* weight3, float* weight4,
	float* input, float* hidden1, float* hidden2, float* hidden3, float* output
);

// +================================================ main 함수 ================================================+
int main() {
	srand((size_t)time(NULL));

	float input[INPUT_NODES];
	float hidden1[HIDDEN1_NODES];
	float hidden2[HIDDEN2_NODES];
	float hidden3[HIDDEN3_NODES];
	float output[OUTPUT_NODES];
	float target[OUTPUT_NODES];
	// Early Stopping 
	float best_accuracy = 0.0f;
	int patience_counter = 0;
	const int patience_limit = 10;
	float epoch_accuracy[EPOCH] = { 0.0f };


	float* weight1 = makeArray(INPUT_NODES, HIDDEN1_NODES);
	float* weight2 = makeArray(HIDDEN1_NODES, HIDDEN2_NODES);
	float* weight3 = makeArray(HIDDEN2_NODES, HIDDEN3_NODES);
	float* weight4 = makeArray(HIDDEN3_NODES, OUTPUT_NODES);

	// 모멘텀 -> velocity 배열 추가
	float* velocity1 = makeArray(INPUT_NODES, HIDDEN1_NODES);
	float* velocity2 = makeArray(HIDDEN1_NODES, HIDDEN2_NODES);
	float* velocity3 = makeArray(HIDDEN2_NODES, HIDDEN3_NODES);
	float* velocity4 = makeArray(HIDDEN3_NODES, OUTPUT_NODES);
	memset(velocity1, 0, sizeof(float) * INPUT_NODES * HIDDEN1_NODES);
	memset(velocity2, 0, sizeof(float) * HIDDEN1_NODES * HIDDEN2_NODES);
	memset(velocity3, 0, sizeof(float) * HIDDEN2_NODES * HIDDEN3_NODES);
	memset(velocity4, 0, sizeof(float) * HIDDEN3_NODES * OUTPUT_NODES);

	initialize_weights(weight1, INPUT_NODES, HIDDEN1_NODES);	// 784 * 64
	printf("weight1[] 랜덤 실수 초기화 완료\n");
	initialize_weights(weight2, HIDDEN1_NODES, HIDDEN2_NODES);	// 64 * 32
	printf("weight2[] 랜덤 실수 초기화 완료\n");
	initialize_weights(weight3, HIDDEN2_NODES, HIDDEN3_NODES);	// 32 * 16
	printf("weight3[] 랜덤 실수 초기화 완료\n");
	initialize_weights(weight4, HIDDEN3_NODES, OUTPUT_NODES);	// 16 * 10
	printf("weight4[] 랜덤 실수 초기화 완료\n");

	for (int epoch = 0; epoch < EPOCH; epoch++) {
		trainNeuralNet(
			input, hidden1, hidden2, hidden3, output, target, (float*)weight1, (float*)weight2, (float*)weight3, (float*)weight4, 
			NUM_HIDDEN_LAYERS, epoch,
			velocity1, velocity2, velocity3, velocity4
		);

		// Early Stopping 판단
		float current_accuracy = test_accuracy(
			ES_TEST_DIR,
			IMG_WIDTH, IMG_HEIGHT, OUTPUT_NODES, NUM_HIDDEN_LAYERS,
			HIDDEN1_NODES, HIDDEN2_NODES, HIDDEN3_NODES,
			weight1, weight2, weight3, weight4,
			input, hidden1, hidden2, hidden3, output
		);
		epoch_accuracy[epoch] = current_accuracy * 100.0f; 
		
		if (current_accuracy > best_accuracy) {
			best_accuracy = current_accuracy;
			patience_counter = 0;

			// 인식률 향상시 SaveMetaData
			saveMetaData(
				hidden1, hidden2, hidden3, output,
				weight1, weight2, weight3, weight4,
				IMG_WIDTH, IMG_HEIGHT, OUTPUT_NODES, NUM_HIDDEN_LAYERS,
				HIDDEN1_NODES, HIDDEN2_NODES, HIDDEN3_NODES,
				RAND_WEIGHT_MIN, RAND_WEIGHT_MAX
			);
		}
		else {
			patience_counter++;
		}

		if (patience_counter >= patience_limit) {
			printf("Early stopping at epoch [%d] 정확도 [%.2f%%]\n", epoch, best_accuracy * 100);
			break;
		}
	}

	printf("\n=== Epoch별 인식률 결과 ===\n");
	for (int i = 0; i < EPOCH; i++) {
		if (epoch_accuracy[i] > 0.0f) {
			printf("Epoch %2d : %.2f%%\n", i + 1, epoch_accuracy[i]);
		}
	}

	free(weight1);
	free(weight2);
	free(weight3);
	free(weight4);
	return 0;
}

// +================================================ 함수 정의부 ================================================+

void softmax(float* input, int length) {
	float max_val = input[0];
	for (int i = 1; i < length; i++) {
		if (input[i] > max_val) {
			max_val = input[i];
		}
	}

	// 1차 변환: (값 - max값) 해서 overflow 방지
	float sum = 0.0f;
	for (int i = 0; i < length; i++) {
		input[i] = expf(input[i] - max_val);  // exp(x - max)
		sum += input[i];
	}

	// 2차 정규화: 전체 합으로 나눔
	for (int i = 0; i < length; i++) {
		input[i] /= sum;
	}
}

float* makeArray(int height, int width) {
	float* arr = (float*)malloc(sizeof(float) * width * height);
	return arr;
}

void trainNeuralNet(
	float* input,
	float* hidden1, float* hidden2, float* hidden3,
	float* output, float* target,
	float* weight1, float* weight2, float* weight3, float* weight4,
	int num_of_hidden_layer, int epoch,
	// 모멘텀 -> velocity 배열
	float* velocity1, float* velocity2, float* velocity3, float* velocity4
) {
	FILE* fp;
	char filepath[256];						// 루프 돌며 파일 경로 저장할 버퍼 배열
	int file_index, folder_index;
	int num_files[OUTPUT_NODES] = { 0 };	// 파일 읽어오다가 fp== NULL 걸리는 순간 인덱스 저장
	int folders_done = 0;					// 파일 입력 중단 플래그

	for (file_index = 0; ; file_index++) {
		folders_done = 0;

		for (folder_index = 0; folder_index < OUTPUT_NODES; folder_index++) {
			// 해당 폴더가 이미 완료된 경우 바로 skip
			if (num_files[folder_index] != 0 && file_index >= num_files[folder_index]) {
				folders_done++;
				continue;
			}

			sprintf(filepath, "%s%d/%d-%d.raw", TRAIN_DIR, folder_index, folder_index, file_index);
			//printf("%s\n", filepath);

			fp = fopen(filepath, "rb");
			if (fp == NULL) {
				printf("학습할 파일이 더 이상 존재하지 않음: %s\n", filepath);
				if (num_files[folder_index] == 0) {
					num_files[folder_index] = file_index - 1; // 첫 실패 때만 저장
				}
				folders_done++;
				continue;
			}

			// 파일 읽기
			unsigned char raw[INPUT_NODES];
			fread(raw, sizeof(unsigned char), INPUT_NODES, fp);

			// 정규화해서 input 배열에 저장
			for (int i = 0; i < INPUT_NODES; i++) {
				input[i] = (float)raw[i] / 255.0f;
				//printf("%.1f ", input[i]);
			}

			printf("EPOCH [%d] 학습 파일 : %s\n", epoch, filepath);
			forwardPropagation(
				input, hidden1, hidden2, hidden3,
				output,
				weight1, weight2, weight3, weight4,
				num_of_hidden_layer
			);

			// target value 지정
			for (int i = 0; i < OUTPUT_NODES; i++) {
				target[i] = (i == folder_index) ? 1.0f : 0.0f;
			}

			backPropagation(
				input, hidden1, hidden2, hidden3,
				output, target,
				weight1, weight2, weight3, weight4,
				num_of_hidden_layer,
				velocity1, velocity2, velocity3, velocity4,
				epoch
			);

			//exit(0);
			fclose(fp);
		}
		// 모든 폴더가 완료되면 즉시 종료
		if (folders_done == OUTPUT_NODES) {
			printf("모든 폴더 읽기 완료\n");
			break;
		}
	}
	for (int i = 0; i < OUTPUT_NODES; i++) {
		printf("마지막 읽어온 파일 : %d/%d.raw\n", i, num_files[i]);
	}
}

void forwardPropagation(
	float* input,
	float* hidden1, float* hidden2, float* hidden3,
	float* output,
	float* weight1, float* weight2, float* weight3, float* weight4,
	int num_of_hidden_layer
) {
	//printf("forwardPropagation() 진입\n");
	switch (num_of_hidden_layer) {
	case 1:
		// input -> hidden layer 1
		for (int i = 0; i < HIDDEN1_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < INPUT_NODES; j++) {
				sum += input[j] * weight1[i * INPUT_NODES + j];
			}
			hidden1[i] = sigmoid(sum);
		}
		// hidden layer 1 -> output
		for (int i = 0; i < OUTPUT_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < HIDDEN1_NODES; j++) {
				sum += hidden1[j] * weight2[i * HIDDEN1_NODES + j];
			}
			output[i] = sum;
		}
		break;
	case 2:
		// input -> hidden layer 1
		for (int i = 0; i < HIDDEN1_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < INPUT_NODES; j++) {
				sum += input[j] * weight1[i * INPUT_NODES + j];
			}
			hidden1[i] = sigmoid(sum);
		}
		// hidden layer 1 -> hidden layer 2
		for (int i = 0; i < HIDDEN2_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < HIDDEN1_NODES; j++) {
				sum += hidden1[j] * weight2[i * HIDDEN1_NODES + j];
			}
			hidden2[i] = sigmoid(sum);
		}
		// hidden layer 2 -> output
		for (int i = 0; i < OUTPUT_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < HIDDEN2_NODES; j++) {
				sum += hidden2[j] * weight3[i * HIDDEN2_NODES + j];
			}
			output[i] = sum;
		}
		break;
	case 3:
		// input -> hidden layer 1
		for (int i = 0; i < HIDDEN1_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < INPUT_NODES; j++) {
				sum += input[j] * weight1[i * INPUT_NODES + j];
			}
			hidden1[i] = sigmoid(sum);
		}
		// hidden layer 1 -> hidden layer 2
		for (int i = 0; i < HIDDEN2_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < HIDDEN1_NODES; j++) {
				sum += hidden1[j] * weight2[i * HIDDEN1_NODES + j];
			}
			hidden2[i] = sigmoid(sum);
		}
		// hidden layer 2 -> hidden layer 3
		for (int i = 0; i < HIDDEN3_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < HIDDEN2_NODES; j++) {
				sum += hidden2[j] * weight3[i * HIDDEN2_NODES + j];
			}
			hidden3[i] = sigmoid(sum);
		}
		// hidden layer 3 -> output
		for (int i = 0; i < OUTPUT_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < HIDDEN3_NODES; j++) {
				sum += hidden3[j] * weight4[i * HIDDEN3_NODES + j];
			}
			output[i] = sum;
		}
		break;
	default:
		printf("올바르지 않은 hidden layer 개수가 전달되었습니다. : %d\n", num_of_hidden_layer);
	}
	softmax(output, OUTPUT_NODES);
}

void backPropagation(
	float* input, float* hidden1, float* hidden2, float* hidden3,
	float* output, float* target,
	float* weight1, float* weight2, float* weight3, float* weight4,
	int num_of_hidden_layer,
	// 모멘텀 -> velocity 배열 받아옴
	float* velocity1, float* velocity2, float* velocity3, float* velocity4,
	int epoch
) {
	float learning_rate = LEARNING_RATE;
	if (epoch > 20) {
		learning_rate *= 0.5f;
	}

	float output_error[OUTPUT_NODES];
	float hidden1_error[HIDDEN1_NODES];
	float hidden2_error[HIDDEN2_NODES];
	float hidden3_error[HIDDEN3_NODES];
	// 1. 출력층 오차 계산
	for (int i = 0; i < OUTPUT_NODES; i++) {
		/*float error = output[i] - target[i];
		output_error[i] = error * sigmoid_derivative(output[i]);*/
		output_error[i] = output[i] - target[i];
	}

	switch (num_of_hidden_layer) {
	case 1:
		// weight2 업데이트 (hidden1 -> output)
		for (int i = 0; i < OUTPUT_NODES; i++) {
			for (int j = 0; j < HIDDEN1_NODES; j++) {
				//weight2[i * HIDDEN1_NODES + j] -= learning_rate * output_error[i] * hidden1[j];

				int idx = i * HIDDEN1_NODES + j;
				float gradient = output_error[i] * hidden1[j];
				velocity2[idx] = MOMENTUM_ALPHA * velocity2[idx] - learning_rate * gradient;
				weight2[idx] += velocity2[idx];
			}
		}

		// hidden1 오차 계산
		for (int i = 0; i < HIDDEN1_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < OUTPUT_NODES; j++) {
				sum += output_error[j] * weight2[j * HIDDEN1_NODES + i];
			}
			hidden1_error[i] = sum * sigmoid_derivative(hidden1[i]);
		}

		// weight1 업데이트 (input -> hidden1)
		for (int i = 0; i < HIDDEN1_NODES; i++) {
			for (int j = 0; j < INPUT_NODES; j++) {
				//weight1[i * INPUT_NODES + j] -= learning_rate * hidden1_error[i] * input[j];

				int idx = i * INPUT_NODES + j;
				float gradient = hidden1_error[i] * input[j];
				velocity1[idx] = MOMENTUM_ALPHA * velocity1[idx] - learning_rate * gradient;
				weight1[idx] += velocity1[idx];
			}
		}
		break;
	case 2:
		// hidden2 오차 계산
		for (int i = 0; i < HIDDEN2_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < OUTPUT_NODES; j++) {
				sum += output_error[j] * weight3[j * HIDDEN2_NODES + i];
			}
			hidden2_error[i] = sum * sigmoid_derivative(hidden2[i]);
		}

		// hidden1 오차 계산
		for (int i = 0; i < HIDDEN1_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < HIDDEN2_NODES; j++) {
				sum += hidden2_error[j] * weight2[j * HIDDEN1_NODES + i];
			}
			hidden1_error[i] = sum * sigmoid_derivative(hidden1[i]);
		}

		// weight3 업데이트 (hidden2 -> output)
		for (int i = 0; i < OUTPUT_NODES; i++) {
			for (int j = 0; j < HIDDEN2_NODES; j++) {
				//weight3[i * HIDDEN2_NODES + j] -= learning_rate * output_error[i] * hidden2[j];

				int idx = i * HIDDEN2_NODES + j;
				float gradient = output_error[i] * hidden2[j];
				velocity3[idx] = MOMENTUM_ALPHA * velocity3[idx] - learning_rate * gradient;
				weight3[idx] += velocity3[idx];
			}
		}

		// weight2 업데이트 (hidden1 -> hidden2)
		for (int i = 0; i < HIDDEN2_NODES; i++) {
			for (int j = 0; j < HIDDEN1_NODES; j++) {
				//weight2[i * HIDDEN1_NODES + j] -= learning_rate * hidden2_error[i] * hidden1[j];

				int idx = i * HIDDEN1_NODES + j;
				float gradient = hidden2_error[i] * hidden1[j];
				velocity2[idx] = MOMENTUM_ALPHA * velocity2[idx] - learning_rate * gradient;
				weight2[idx] += velocity2[idx];
			}
		}

		// weight1 업데이트 (input -> hidden1)
		for (int i = 0; i < HIDDEN1_NODES; i++) {
			for (int j = 0; j < INPUT_NODES; j++) {
				//weight1[i * INPUT_NODES + j] -= learning_rate * hidden1_error[i] * input[j];

				int idx = i * INPUT_NODES + j;
				float gradient = hidden1_error[i] * input[j];
				velocity1[idx] = MOMENTUM_ALPHA * velocity1[idx] - learning_rate * gradient;
				weight1[idx] += velocity1[idx];
			}
		}
		break;
	case 3:
		// hidden3 오차 계산
		for (int i = 0; i < HIDDEN3_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < OUTPUT_NODES; j++) {
				sum += output_error[j] * weight4[j * HIDDEN3_NODES + i];
			}
			hidden3_error[i] = sum * sigmoid_derivative(hidden3[i]);
		}

		// hidden2 오차 계산
		for (int i = 0; i < HIDDEN2_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < HIDDEN3_NODES; j++) {
				sum += hidden3_error[j] * weight3[j * HIDDEN2_NODES + i];
			}
			hidden2_error[i] = sum * sigmoid_derivative(hidden2[i]);
		}

		// hidden1 오차 계산
		for (int i = 0; i < HIDDEN1_NODES; i++) {
			float sum = 0.0f;
			for (int j = 0; j < HIDDEN2_NODES; j++) {
				sum += hidden2_error[j] * weight2[j * HIDDEN1_NODES + i];
			}
			hidden1_error[i] = sum * sigmoid_derivative(hidden1[i]);
		}

		// weight4 업데이트 (hidden3 -> output)
		for (int i = 0; i < OUTPUT_NODES; i++) {
			for (int j = 0; j < HIDDEN3_NODES; j++) {
				//weight4[i * HIDDEN3_NODES + j] -= learning_rate * output_error[i] * hidden3[j];

				int idx = i * HIDDEN3_NODES + j;
				float gradient = output_error[i] * hidden3[j];
				velocity4[idx] = MOMENTUM_ALPHA * velocity4[idx] - learning_rate * gradient;
				weight4[idx] += velocity4[idx];
			}
		}

		// weight3 업데이트 (hidden2 -> hidden3)
		for (int i = 0; i < HIDDEN3_NODES; i++) {
			for (int j = 0; j < HIDDEN2_NODES; j++) {
				//weight3[i * HIDDEN2_NODES + j] -= learning_rate * hidden3_error[i] * hidden2[j];

				int idx = i * HIDDEN2_NODES + j;
				float gradient = hidden3_error[i] * hidden2[j];
				velocity3[idx] = MOMENTUM_ALPHA * velocity3[idx] - learning_rate * gradient;
				weight3[idx] += velocity3[idx];
			}
		}

		// weight2 업데이트 (hidden1 -> hidden2)
		for (int i = 0; i < HIDDEN2_NODES; i++) {
			for (int j = 0; j < HIDDEN1_NODES; j++) {
				//weight2[i * HIDDEN1_NODES + j] -= learning_rate * hidden2_error[i] * hidden1[j];

				int idx = i * HIDDEN1_NODES + j;
				float gradient = hidden2_error[i] * hidden1[j];
				velocity2[idx] = MOMENTUM_ALPHA * velocity2[idx] - learning_rate * gradient;
				weight2[idx] += velocity2[idx];
			}
		}

		// weight1 업데이트 (input -> hidden1)
		for (int i = 0; i < HIDDEN1_NODES; i++) {
			for (int j = 0; j < INPUT_NODES; j++) {
				//weight1[i * INPUT_NODES + j] -= learning_rate * hidden1_error[i] * input[j];

				int idx = i * INPUT_NODES + j;
				float gradient = hidden1_error[i] * input[j];
				velocity1[idx] = MOMENTUM_ALPHA * velocity1[idx] - learning_rate * gradient;
				weight1[idx] += velocity1[idx];
			}
		}
		break;
	default:
		printf("올바르지 않은 hidden layer 개수입니다 : %d\n", num_of_hidden_layer);
	}
}

/**
 * @brief 지정된 범위 내의 난수로 가중치 배열을 초기화하는 함수
 *
 * 각 weight[i][j] = RAND_WEIGHT_MIN + rand()를 통해 RAND_WEIGHT_MIN에서 RAND_WEIGHT_MAX 사이의 float값으로 초기화됨.
 * 배열은 1차원으로 표현된 2차원 형태로 저장되며, 행은 right(출력 노드 수), 열은 left(입력 노드 수)로 구성됨.
 *
 * @param weight 가중치 배열의 포인터 (크기: right * left)
 * @param left 이전 층 노드 수 (입력 차원 수)
 * @param right 현재 층 노드 수 (출력 차원 수)
 */
void initialize_weights(float* weight, int left, int right) {
	for (int i = 0; i < right; i++) { // 64
		for (int j = 0; j < left; j++) { // 784
			weight[i * left + j] = RAND_WEIGHT_MIN + ((float)rand() / RAND_MAX) * (RAND_WEIGHT_MAX - RAND_WEIGHT_MIN);
			//printf("%.1f ", weight[i * left + j]);
		}
	}
}

float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
	return x * (1.0f - x);
}

void saveMetaData(
	float* hidden1, float* hidden2, float* hidden3,
	float* output, float* weight1, float* weight2, float* weight3, float* weight4,
	int img_width, int img_height, int output_nodes, int num_hidden_layer,
	int hidden1_nodes, int hidden2_nodes, int hidden3_nodes,
	float rand_weight_min, float rand_weight_max
) {
	//printf("파일 입력 시작\n");

	char filepath[256];
	sprintf(filepath, "%s_h%d.txt", META_DIR, NUM_HIDDEN_LAYERS);

	FILE* fp = fopen(filepath, "w");

	fprintf(fp, "#IMG_WIDTH=%d\n", img_width);
	fprintf(fp, "#IMG_HEIGHT=%d\n", img_height);
	fprintf(fp, "#OUTPUT_NODES=%d\n", output_nodes);
	fprintf(fp, "#NUM_HIDDEN_LAYER=%d\n", num_hidden_layer);
	fprintf(fp, "#HIDDEN1_NODES=%d\n", hidden1_nodes);
	fprintf(fp, "#HIDDEN2_NODES=%d\n", hidden2_nodes);
	fprintf(fp, "#HIDDEN3_NODES=%d\n", hidden3_nodes);
	fprintf(fp, "#RAND_WEIGHT_MIN=%f\n", rand_weight_min);
	fprintf(fp, "#RAND_WEIGHT_MAX=%f\n", rand_weight_max);


	// 가중치 저장 -> hidden layer 개수에 따라 다르게 저장
	// input -> hidden1
	write_array(fp, "WEIGHT1", weight1, INPUT_NODES * HIDDEN1_NODES);

	if (num_hidden_layer == 1) {
		// hidden1 -> output
		write_array(fp, "WEIGHT2", weight2, HIDDEN1_NODES * OUTPUT_NODES);
	}
	else if (num_hidden_layer == 2) {
		// hidden1 -> hidden2
		write_array(fp, "WEIGHT2", weight2, HIDDEN1_NODES * HIDDEN2_NODES);
		// hidden2 -> output
		write_array(fp, "WEIGHT3", weight3, HIDDEN2_NODES * OUTPUT_NODES);
	}
	else if (num_hidden_layer == 3) {
		// hidden1 -> hidden2
		write_array(fp, "WEIGHT2", weight2, HIDDEN1_NODES * HIDDEN2_NODES);
		// hidden2 -> hidden3
		write_array(fp, "WEIGHT3", weight3, HIDDEN2_NODES * HIDDEN3_NODES);
		// hidden3 -> output
		write_array(fp, "WEIGHT4", weight4, HIDDEN3_NODES * OUTPUT_NODES);
	}

	/*write_array(fp, "HIDDEN1", hidden1, HIDDEN1_NODES);

	if (num_hidden_layer >= 2) {
		write_array(fp, "HIDDEN2", hidden2, HIDDEN2_NODES);
	}
	if (num_hidden_layer == 3) {
		write_array(fp, "HIDDEN3", hidden3, HIDDEN3_NODES);
	}*/

	write_array(fp, "OUTPUT", output, OUTPUT_NODES);

	fclose(fp);
	printf("%s 저장 완료\n", filepath);
}

void write_array(FILE* fp, const char* name, float* arr, int size) {
	fprintf(fp, "[%s]\n", name);
	for (int i = 0; i < size; i++) {
		fprintf(fp, "%f", arr[i]);
		if (i != size - 1) fprintf(fp, ",");
		if ((i + 1) % 16 == 0) fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");
}

float test_accuracy(
	const char* test_dir,
	int img_width, int img_height, int output_nodes, int num_hidden_layer,
	int hidden1_nodes, int hidden2_nodes, int hidden3_nodes,
	float* weight1, float* weight2, float* weight3, float* weight4,
	float* input, float* hidden1, float* hidden2, float* hidden3, float* output
) {
	char filepath[256];
	int correct = 0;
	int total = 0;
	int num_files[10] = { 0 };
	int folders_done;
	unsigned char* raw = (unsigned char*)malloc(sizeof(unsigned char) * img_width * img_height);

	if (!raw) {
		printf("raw malloc 실패\n");
		return 0.0f;
	}

	for (int file_index = 0; ; file_index++) {
		folders_done = 0;

		for (int folder_index = 0; folder_index < output_nodes; folder_index++) {
			if (num_files[folder_index] != 0 && file_index >= num_files[folder_index]) {
				folders_done++;
				continue;
			}

			sprintf(filepath, "%s%d/%d-%d.raw", test_dir, folder_index, folder_index, file_index);

			FILE* fp = fopen(filepath, "rb");
			if (!fp) {
				if (num_files[folder_index] == 0) {
					num_files[folder_index] = file_index;
				}
				folders_done++;
				continue;
			}

			fread(raw, sizeof(unsigned char), img_width * img_height, fp);
			fclose(fp);

			for (int i = 0; i < img_width * img_height; i++) {
				input[i] = (float)raw[i] / 255.0f;
			}

			forwardPropagation(
				input, hidden1, hidden2, hidden3, output,
				weight1, weight2, weight3, weight4,
				num_hidden_layer,
				img_width * img_height, output_nodes,
				hidden1_nodes, hidden2_nodes, hidden3_nodes
			);

			int predicted = 0;
			float max_val = output[0];
			for (int i = 1; i < output_nodes; i++) {
				if (output[i] > max_val) {
					max_val = output[i];
					predicted = i;
				}
			}

			if (predicted == folder_index) {
				correct++;
			}
			total++;
		}

		if (folders_done == output_nodes) {
			break;
		}
	}

	free(raw);
	return (float)correct / total;
}

