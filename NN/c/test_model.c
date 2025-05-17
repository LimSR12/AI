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
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <windows.h>

#define NUM_HIDDEN_LAYERS 2
#define TEST_DIR "../../mnist_raw/testing/"
#define META_DIR "../../meta"
#define MOMENTUM_META_DIR "../../meta_momentum"
#define SOFTMAX_META_DIR "../../meta_sotmax_crossentropy"
#define EARLY_STOPPING_META_DIR "../../meta_earlystopping"

/*
* main() 함수에서 readMetadata()와 loadWeightsFromMetadata()에 전달하는 META_DIR을 변경해주면 기존의 메타데이터를 통해 테스트를 진행할 수 있습니다.
* META_DIR                      default
* MOMENTUM_META_DIR             모멘텀 추가한 학습모델 메타데이터
* SOFTMAX_META_DIR              소프트맥스+교차엔트로피 추가한 학습모델 메타데이터
* EARLY_STOPPING_META_DIR       early stopping까지 추가한 학습모델 메타데이터
*/

#define MAX_LINE 16384

// +================================================ 함수 선언부 ================================================+

float* makeArray(int size);

void readMetadata(
    int* img_width, int* img_height, int* output_nodes, int* num_hidden_layer,
    int* hidden1_nodes, int* hidden2_nodes, int* hidden3_nodes,
    float* rand_min, float* rand_max,
    const char* path
);
void loadArrayFromMetadata(const char* tag, float* array, int expected_size, const char* path);

void loadWeightsFromMetadata(
    int input_nodes, int output_nodes,
    int num_hidden_layer,
    int hidden1_nodes, int hidden2_nodes, int hidden3_nodes,
    float** weight1, float** weight2, float** weight3, float** weight4,
    const char* path
);
void forwardPropagation(
    float* input,
    float* hidden1, float* hidden2, float* hidden3,
    float* output,
    float* weight1, float* weight2, float* weight3, float* weight4,
    int num_hidden_layer,
    int input_nodes, int output_nodes,
    int hidden1_nodes, int hidden2_nodes, int hidden3_nodes
);

void printResult(float* output, int output_nodes);

float sigmoid(float x);

void testOneImage(
    const char* filename,
    int img_width, int img_height,
    int output_nodes, int num_hidden_layer,
    int hidden1_nodes, int hidden2_nodes, int hidden3_nodes,
    float* input, float* hidden1, float* hidden2, float* hidden3, float* output,
    float* weight1, float* weight2, float* weight3, float* weight4
);

void testAllImages(
    const char* filename,
    int img_width, int img_height,
    int output_nodes, int num_hidden_layer,
    int hidden1_nodes, int hidden2_nodes, int hidden3_nodes,
    float* input, float* hidden1, float* hidden2, float* hidden3, float* output,
    float* weight1, float* weight2, float* weight3, float* weight4
);

// +================================================ main 함수 ================================================+
int main() {
    // 메타데이터로부터 읽을 변수들
    int IMG_WIDTH = 0, IMG_HEIGHT = 0;
    int OUTPUT_NODES = 0;
    int NUM_HIDDEN_LAYER = 0;
    int HIDDEN1_NODES = 0, HIDDEN2_NODES = 0, HIDDEN3_NODES = 0;
    float RAND_WEIGHT_MIN = 0.0f, RAND_WEIGHT_MAX = 0.0f;

    // 메타데이터로부터 읽어온 실수값 저장할 배열들
    float* weight1 = NULL;
    float* weight2 = NULL;
    float* weight3 = NULL;
    float* weight4 = NULL;

    float* input = NULL;
    float* hidden1 = NULL;
    float* hidden2 = NULL;
    float* hidden3 = NULL;
    float* output = NULL;

    // 메타데이터로부터 상수 읽어와서 저장
    readMetadata(
        &IMG_WIDTH, &IMG_HEIGHT, &OUTPUT_NODES, &NUM_HIDDEN_LAYER,
        &HIDDEN1_NODES, &HIDDEN2_NODES, &HIDDEN3_NODES,
        &RAND_WEIGHT_MIN, &RAND_WEIGHT_MAX,
        META_DIR
        //MOMENTUM_META_DIR
        //SOFTMAX_META_DIR
        //EARLY_STOPPING_META_DIR
    );

    // 메타데이터로부터 가중치 읽어와서 넘겨준 배열에 저장
    loadWeightsFromMetadata(
        IMG_WIDTH * IMG_HEIGHT, OUTPUT_NODES,
        NUM_HIDDEN_LAYER,
        HIDDEN1_NODES, HIDDEN2_NODES, HIDDEN3_NODES,
        &weight1, &weight2, &weight3, &weight4,
        META_DIR
        //MOMENTUM_META_DIR
        //SOFTMAX_META_DIR
        //EARLY_STOPPING_META_DIR
    );

    input = makeArray(IMG_WIDTH * IMG_HEIGHT);
    hidden1 = makeArray(HIDDEN1_NODES);
    hidden2 = makeArray(HIDDEN2_NODES);
    hidden3 = makeArray(HIDDEN3_NODES);
    output = makeArray(OUTPUT_NODES);


    /*testOneImage(
        TEST_DIR,
        IMG_WIDTH, IMG_HEIGHT,
        OUTPUT_NODES, NUM_HIDDEN_LAYER,
        HIDDEN1_NODES, HIDDEN2_NODES, HIDDEN3_NODES,
        input, hidden1, hidden2, hidden3, output,
        weight1, weight2, weight3, weight4
    );*/

    // 모든 테스트 이미지 실행
    testAllImages(
        TEST_DIR,
        IMG_WIDTH, IMG_HEIGHT, OUTPUT_NODES, NUM_HIDDEN_LAYER,
        HIDDEN1_NODES, HIDDEN2_NODES, HIDDEN3_NODES,
        weight1, weight2, weight3, weight4,
        input, hidden1, hidden2, hidden3, output
    );

    free(weight1);
    if (weight2) free(weight2);
    if (weight3) free(weight3);
    if (weight4) free(weight4);

    free(input);
    free(hidden1);
    free(hidden2);
    free(hidden3);
    free(output);

    return 0;
}

// +================================================ 함수 정의부 ================================================+
/**
 * @brief size 길이의 배열 동적할당 후 주소 반환
 * @param size 배열 길이
 */
float* makeArray(int size) {
    float* arr = (float*)malloc(sizeof(float) * size);
    return arr;
}

/**
 * @brief 문자열의 앞뒤 공백을 제거
 * @param str 대상 문자열
 */
void trim(char* str) {
    /*
        [앞]
        문자열의 첫 글자가 공백 문자(띄어쓰기, 탭, 줄바꿈 등)인지 확인하면서 포인터를 다음 문자로 이동.
        이 과정을 통해 문자열의 시작 위치가 공백이 아닌 첫 유효 문자로 이동함.
        [뒤]
        문자열 끝에서부터 isspace()로 공백 문자인지 검사하면서 뒤로 이동.
        공백이 아닌 문자를 만나면 거기서 멈춤.
        [예시]
        char str[] = "   \t Hello World!  \n";
        trim(str);
        printf("[%s]", str);   // -> 출력: Hello World!
    */
    while (isspace((unsigned char)*str)) str++; // 앞쪽 공백 제거

    char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--; // 뒤쪽 공백 제거

    *(end + 1) = '\0';
}

/**
 * @brief 메타데이터 파일에서 전체 구조 정보를 읽어오는 함수
 *
 * @param img_width 이미지 너비
 * @param img_height 이미지 높이
 * @param output_nodes 출력 노드 개수
 * @param num_hidden_layer hidden layer 개수
 * @param hidden1_nodes 첫 번째 hidden layer 노드 수
 * @param hidden2_nodes 두 번째 hidden layer 노드 수
 * @param hidden3_nodes 세 번째 hidden layer 노드 수
 * @param rand_min 초기 가중치 최소값
 * @param rand_max 초기 가중치 최대값
 */
void readMetadata(
    int* img_width, int* img_height, int* output_nodes, int* num_hidden_layer,
    int* hidden1_nodes, int* hidden2_nodes, int* hidden3_nodes,
    float* rand_min, float* rand_max,
    const char* path
) {
    char filepath[256];
    sprintf(filepath, "%s_h%d.txt", path, NUM_HIDDEN_LAYERS);
    //printf("%s\n", filepath);
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL) {
        printf("파일이 존재하지 않습니다 : %s\n", filepath);
        printf("[Warning]학습 모델을 먼저 실행하여 메타데이터 파일을 생성해야 합니다.\n");
        printf("[Warning]디렉토리 구조를 확인해주세요.\n");
        printf("[Warning]현재 NUM_HIDDEN_LAYERS 값 : %d\n", NUM_HIDDEN_LAYERS);
        exit(1);
    }

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        trim(line);

        // 매크로 상수만 필터링: '# '로 시작하는 줄
        if (line[0] == '#') {
            char name[64];
            float value;

            if (sscanf(line, "# %[^=]= %f", name, &value) == 2 || sscanf(line, "# %[^=] = %f", name, &value) == 2) {
                trim(name);
                if (strcmp(name, "RAND_WEIGHT_MIN") == 0)
                    *(rand_min) = value;
                else if (strcmp(name, "RAND_WEIGHT_MAX") == 0)
                    *(rand_max) = value;
                else if (strcmp(name, "IMG_WIDTH") == 0)
                    *(img_width) = value;
                else if (strcmp(name, "IMG_HEIGHT") == 0)
                    *(img_height) = value;
                else if (strcmp(name, "OUTPUT_NODES") == 0)
                    *(output_nodes) = value;
                else if (strcmp(name, "NUM_HIDDEN_LAYER") == 0)
                    *(num_hidden_layer) = value;
                else if (strcmp(name, "HIDDEN1_NODES") == 0)
                    *(hidden1_nodes) = value;
                else if (strcmp(name, "HIDDEN2_NODES") == 0)
                    *(hidden2_nodes) = value;
                else if (strcmp(name, "HIDDEN3_NODES") == 0)
                    *(hidden3_nodes) = value;
            }
            else {
                printf("[형식 오류] %s\n", line);
            }
        }
    }
    printf("%s 로부터 읽어온 데이터\n", filepath);
    printf("IMG_WIDTH = %d\nIMG_HEIGHT = %d\n", *(img_width), *(img_height));
    printf("NUM_HIDDEN_LAYER = %d\n", *(num_hidden_layer));
    printf("HIDDEN1_NODES = %d\nHIDDEN2_NODES = %d\nHIDDEN3_NODES = %d\n", *(hidden1_nodes), *(hidden2_nodes), *(hidden3_nodes));
    printf("RAND_WEIGHT_MIN = %.2f\nRAND_WEIGHT_MAX = %.2f\n", *(rand_min), *(rand_max));

    //exit(0);
    fclose(fp);
}

/**
 * @brief 메타데이터에서 특정 태그의 배열 데이터를 읽어오는 함수
 *
 * @param tag 태그 이름 (예: "WEIGHT1")
 * @param array 읽은 데이터를 저장할 배열
 * @param expected_size 기대되는 배열 크기 (검증용)
 */
void loadArrayFromMetadata(const char* tag, float* array, int expected_size, const char* path) {
    char filepath[256];
    sprintf(filepath, "%s_h%d.txt", path, NUM_HIDDEN_LAYERS);
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL) {
        printf("파일이 존재하지 않습니다 : %s\n", filepath);
        printf("[Warning]학습 모델을 먼저 실행하여 메타데이터 파일을 생성해야 합니다.\n");
        printf("[Warning]디렉토리 구조를 확인해주세요.\n");
        printf("[Warning]현재 NUM_HIDDEN_LAYERS 값 : %d\n", NUM_HIDDEN_LAYERS);
        exit(1);
    }

    char line[MAX_LINE];
    char current_section[64] = "";
    int found = 0;
    int index = 0;

    while (fgets(line, sizeof(line), fp)) {
        trim(line);

        if (line[0] == '\0') continue;

        if (line[0] == '[') {
            sscanf(line, "[%[^]]", current_section);
            found = (strcmp(current_section, tag) == 0);
            continue;
        }

        if (found) {
            char* token = strtok(line, ",");
            while (token != NULL && index < expected_size) {
                array[index++] = atof(token);
                token = strtok(NULL, ",");
            }
        }

        if (found && index >= expected_size) break;
    }

    printf("[%s] : 기대 %d개, 실제 %d개\n", tag, expected_size, index);

    fclose(fp);
}

void loadWeightsFromMetadata(
    int input_nodes, int output_nodes,
    int num_hidden_layer,
    int hidden1_nodes, int hidden2_nodes, int hidden3_nodes,
    float** weight1, float** weight2, float** weight3, float** weight4,
    const char* path
) {
    *weight1 = makeArray(input_nodes * hidden1_nodes);
    loadArrayFromMetadata("WEIGHT1", *weight1, input_nodes * hidden1_nodes, path);

    switch (num_hidden_layer) {
    case 1:
        *weight2 = makeArray(hidden1_nodes * output_nodes);
        loadArrayFromMetadata("WEIGHT2", *weight2, hidden1_nodes * output_nodes, path);
        break;

    case 2:
        *weight2 = makeArray(hidden1_nodes * hidden2_nodes);
        loadArrayFromMetadata("WEIGHT2", *weight2, hidden1_nodes * hidden2_nodes, path);

        *weight3 = makeArray(hidden2_nodes * output_nodes);
        loadArrayFromMetadata("WEIGHT3", *weight3, hidden2_nodes * output_nodes, path);
        break;

    case 3:
        *weight2 = makeArray(hidden1_nodes * hidden2_nodes);
        loadArrayFromMetadata("WEIGHT2", *weight2, hidden1_nodes * hidden2_nodes, path);

        *weight3 = makeArray(hidden2_nodes * hidden3_nodes);
        loadArrayFromMetadata("WEIGHT3", *weight3, hidden2_nodes * hidden3_nodes, path);

        *weight4 = makeArray(hidden3_nodes * output_nodes);
        loadArrayFromMetadata("WEIGHT4", *weight4, hidden3_nodes * output_nodes, path);
        break;

    default:
        printf("올바르지 않은 hidden layer 개수가 전달되었습니다. : %d\n", num_hidden_layer);
        break;
    }
}


/**
 * @brief 신경망의 forward propagation을 수행하여 출력값을 계산하는 함수
 *
 * @param input 입력 배열
 * @param hidden1 첫 번째 hidden layer 배열
 * @param hidden2 두 번째 hidden layer 배열
 * @param hidden3 세 번째 hidden layer 배열
 * @param output 출력 배열
 * @param weight1 input -> hidden1 가중치 배열
 * @param weight2 hidden1 -> hidden2 가중치 배열 // num_hidden_layer == 1 이면 얘가 output으로
 * @param weight3 hidden2 -> hidden3 가중치 배열 // num_hidden_layer == 2 이면 얘가 output으로
 * @param weight4 hidden3 -> output 가중치 배열  // num_hidden_layer == 3 이면 얘가 output으로
 *
 * @param num_hidden_layer hidden layer 개수 (1~3)
 * @param input_nodes 입력 노드 수
 * @param output_nodes 출력 노드 수
 * @param hidden1_nodes 첫 번째 hidden layer 노드 수
 * @param hidden2_nodes 두 번째 hidden layer 노드 수
 * @param hidden3_nodes 세 번째 hidden layer 노드 수
 */
void forwardPropagation(
    float* input,
    float* hidden1, float* hidden2, float* hidden3,
    float* output,
    float* weight1, float* weight2, float* weight3, float* weight4,
    int num_hidden_layer,
    int input_nodes, int output_nodes,
    int hidden1_nodes, int hidden2_nodes, int hidden3_nodes
) {

    //printf("forwardPropagation() 진입\n");

    switch (num_hidden_layer) {
    case 1:
        // input -> hidden layer 1
        for (int i = 0; i < hidden1_nodes; i++) {
            float sum = 0.0f;
            for (int j = 0; j < input_nodes; j++) {
                sum += input[j] * weight1[i * input_nodes + j];
            }
            hidden1[i] = sigmoid(sum);
        }

        // hidden layer 1 -> output
        for (int i = 0; i < output_nodes; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden1_nodes; j++) {
                sum += hidden1[j] * weight2[i * hidden1_nodes + j];
            }
            output[i] = sigmoid(sum);
        }
        break;
    case 2:
        // input -> hidden layer 1
        for (int i = 0; i < hidden1_nodes; i++) {
            float sum = 0.0f;
            for (int j = 0; j < input_nodes; j++) {
                sum += input[j] * weight1[i * input_nodes + j];
            }
            hidden1[i] = sigmoid(sum);
        }
        // hidden layer 1 -> hidden layer 2
        for (int i = 0; i < hidden2_nodes; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden1_nodes; j++) {
                sum += hidden1[j] * weight2[i * hidden1_nodes + j];
            }
            hidden2[i] = sigmoid(sum);
        }
        // hidden layer 2 -> output
        for (int i = 0; i < output_nodes; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden2_nodes; j++) {
                sum += hidden2[j] * weight3[i * hidden2_nodes + j];
            }
            output[i] = sigmoid(sum);
        }
        break;
    case 3:
        // input -> hidden layer 1
        for (int i = 0; i < hidden1_nodes; i++) {
            float sum = 0.0f;
            for (int j = 0; j < input_nodes; j++) {
                sum += input[j] * weight1[i * input_nodes + j];
            }
            hidden1[i] = sigmoid(sum);
        }
        // hidden layer 1 -> hidden layer 2
        for (int i = 0; i < hidden2_nodes; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden1_nodes; j++) {
                sum += hidden1[j] * weight2[i * hidden1_nodes + j];
            }
            hidden2[i] = sigmoid(sum);
        }
        // hidden layer 2 -> hidden layer 3
        for (int i = 0; i < hidden3_nodes; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden2_nodes; j++) {
                sum += hidden2[j] * weight3[i * hidden2_nodes + j];
            }
            hidden3[i] = sigmoid(sum);
        }
        // hidden layer 3 -> output
        for (int i = 0; i < output_nodes; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden3_nodes; j++) {
                sum += hidden3[j] * weight4[i * hidden3_nodes + j];
            }
            output[i] = sigmoid(sum);
        }
        break;
    default:
        printf("올바르지 않은 hidden layer 개수가 전달되었습니다. : %d\n", num_hidden_layer);
    }
}

/**
 * @brief 출력값 중 가장 높은 값을 찾아 예측 결과를 출력하는 함수
 *
 * @param output 출력 노드 배열
 * @param output_nodes 출력 노드 개수
 */
void printResult(float* output, int output_nodes) {
    int recog = 0;
    float max_val = output[0];

    for (int i = 1; i < output_nodes; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            recog = i;
        }
    }

    printf("인식 결과: %d (결과값: %.3f)\n", recog, max_val);
}

/**
 * @brief 시그모이드 활성화 함수
 *
 * @param x 입력값
 * @return float 시그모이드 결과
 */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void testOneImage(
    const char* filename,
    int img_width, int img_height,
    int output_nodes, int num_hidden_layer,
    int hidden1_nodes, int hidden2_nodes, int hidden3_nodes,
    float* input, float* hidden1, float* hidden2, float* hidden3, float* output,
    float* weight1, float* weight2, float* weight3, float* weight4
) {
    FILE* image_fp = fopen(filename, "rb");
    if (!image_fp) {
        printf("이미지 파일 열기 실패: %s\n", filename);
        return;
    }

    unsigned char* raw = (unsigned char*)malloc(sizeof(unsigned char) * img_width * img_height);
    fread(raw, sizeof(unsigned char), img_width * img_height, image_fp);
    fclose(image_fp);

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

    printResult(output, output_nodes);

    free(raw);
};

void testAllImages(
    const char* filename,
    int IMG_WIDTH, int IMG_HEIGHT, int OUTPUT_NODES, int NUM_HIDDEN_LAYER,
    int HIDDEN1_NODES, int HIDDEN2_NODES, int HIDDEN3_NODES,
    float* weight1, float* weight2, float* weight3, float* weight4,
    float* input, float* hidden1, float* hidden2, float* hidden3, float* output
) {
    char filepath[256];                     // 파일명 버퍼
    int correct = 0;                        // 맞춘 개수
    int total = 0;                          // 전체 시도 개수
    int correct_per_folder[10] = { 0 };     // 폴더별 맞춘 개수
    int total_per_folder[10] = { 0 };       // 폴더별 전체 시도 개수
    int num_files[10] = { 0 };              // 폴더별 완료 인덱스 저장 배열
    int folders_done;                       // 반복 종료 플래그

    unsigned char* raw = (unsigned char*)malloc(sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);

    if (!raw) {
        printf("raw 배열 malloc 실패\n");
        return;
    }
    for (int file_index = 0; ; file_index++) {
        folders_done = 0;

        for (int folder_index = 0; folder_index < OUTPUT_NODES; folder_index++) {
            // 해당 폴더가 이미 완료된 경우 바로 skip
            if (num_files[folder_index] != 0 && file_index >= num_files[folder_index]) {
                folders_done++;
                continue;
            }
            // 파일 경로 구성
            sprintf(filepath, "%s%d/%d-%d.raw", filename, folder_index, folder_index, file_index);

            //printf("filepath:%s\n", filepath);
            FILE* fp = fopen(filepath, "rb");
            if (!fp) {
                printf("인식할 파일이 더 이상 존재하지 않음: %s\n", filepath);
                if (num_files[folder_index] == 0) {
                    num_files[folder_index] = file_index;
                    total_per_folder[folder_index] = file_index;
                }
                folders_done++;
                continue;
            }

            fread(raw, sizeof(unsigned char), IMG_WIDTH * IMG_HEIGHT, fp);
            fclose(fp);

            // 정규화
            for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
                input[i] = (float)raw[i] / 255.0f;
            }

            forwardPropagation(
                input, hidden1, hidden2, hidden3, output,
                weight1, weight2, weight3, weight4,
                NUM_HIDDEN_LAYER,
                IMG_WIDTH * IMG_HEIGHT, OUTPUT_NODES,
                HIDDEN1_NODES, HIDDEN2_NODES, HIDDEN3_NODES
            );

            // 예측값 찾기
            int predicted = 0;
            float max_val = output[0];
            for (int i = 1; i < OUTPUT_NODES; i++) {
                if (output[i] > max_val) {
                    max_val = output[i];
                    predicted = i;
                }
            }
            // 결과 출력
            printf("[인식 파일: %s] -> 예측: %d | 실제: %d %s\n", filepath, predicted, folder_index, predicted == folder_index ? "O" : "X");

            if (predicted == folder_index) {
                correct++;
                correct_per_folder[folder_index]++;
            }

            total++;
        }
        // 모든 폴더가 완료되면 즉시 종료
        if (folders_done == OUTPUT_NODES) {
            printf("모든 폴더 읽기 완료\n");
            break;
        }
    }

    free(raw);
    float accuracy = (float)correct / total * 100.0f;

    printf("+---------------------------------------+\n");
    printf("전체 인식률: %.2f%% (%d / %d)\n", accuracy, correct, total);
    printf("+---------------------------------------+\n");
    printf("| 숫자 |  전체   |  맞춤   | 인식률(%%)  |\n");
    printf("+------+---------+---------+------------+\n");

    for (int i = 0; i < OUTPUT_NODES; i++) {
        int total_count = total_per_folder[i];
        int correct_count = correct_per_folder[i];
        float acc = total_count > 0 ? (float)correct_count / total_count * 100.0f : 0.0f;

        printf("|  %2d  |  %5d  |  %5d  |   %6.2f%%  |\n", i, total_count, correct_count, acc);
    }
    printf("+---------------------------------------+\n");


}