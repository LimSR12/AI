/*
directory architecture
����mnist_raw
��  ����testing
��  ��  ����0
��  ��  ����1
��  ��  ����2
��  ��  ����3
��  ��  ����4
��  ��  ����5
��  ��  ����6
��  ��  ����7
��  ��  ����8
��  ��  ����9
��  ����training
��      ����0
��      ����1
��      ����2
��      ����3
��      ����4
��      ����5
��      ����6
��      ����7
��      ����8
��      ����9
����neural_net_model
��  ����neural_net_model
��  ��  ����x64
��  ��      ����Debug
��  ��          ����neural_net_model.tlog
��  ����x64
��      ����Debug
����neural_net_testing_model
    ����neural_net_testing_model
    ��  ����x64
    ��      ����Debug
    ��          ����neural_n.633f50e4.tlog
    ����x64
        ����Debug
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
* main() �Լ����� readMetadata()�� loadWeightsFromMetadata()�� �����ϴ� META_DIR�� �������ָ� ������ ��Ÿ�����͸� ���� �׽�Ʈ�� ������ �� �ֽ��ϴ�.
* META_DIR                      default
* MOMENTUM_META_DIR             ����� �߰��� �н��� ��Ÿ������
* SOFTMAX_META_DIR              ����Ʈ�ƽ�+������Ʈ���� �߰��� �н��� ��Ÿ������
* EARLY_STOPPING_META_DIR       early stopping���� �߰��� �н��� ��Ÿ������
*/

#define MAX_LINE 16384

// +================================================ �Լ� ����� ================================================+

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

// +================================================ main �Լ� ================================================+
int main() {
    // ��Ÿ�����ͷκ��� ���� ������
    int IMG_WIDTH = 0, IMG_HEIGHT = 0;
    int OUTPUT_NODES = 0;
    int NUM_HIDDEN_LAYER = 0;
    int HIDDEN1_NODES = 0, HIDDEN2_NODES = 0, HIDDEN3_NODES = 0;
    float RAND_WEIGHT_MIN = 0.0f, RAND_WEIGHT_MAX = 0.0f;

    // ��Ÿ�����ͷκ��� �о�� �Ǽ��� ������ �迭��
    float* weight1 = NULL;
    float* weight2 = NULL;
    float* weight3 = NULL;
    float* weight4 = NULL;

    float* input = NULL;
    float* hidden1 = NULL;
    float* hidden2 = NULL;
    float* hidden3 = NULL;
    float* output = NULL;

    // ��Ÿ�����ͷκ��� ��� �о�ͼ� ����
    readMetadata(
        &IMG_WIDTH, &IMG_HEIGHT, &OUTPUT_NODES, &NUM_HIDDEN_LAYER,
        &HIDDEN1_NODES, &HIDDEN2_NODES, &HIDDEN3_NODES,
        &RAND_WEIGHT_MIN, &RAND_WEIGHT_MAX,
        META_DIR
        //MOMENTUM_META_DIR
        //SOFTMAX_META_DIR
        //EARLY_STOPPING_META_DIR
    );

    // ��Ÿ�����ͷκ��� ����ġ �о�ͼ� �Ѱ��� �迭�� ����
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

    // ��� �׽�Ʈ �̹��� ����
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

// +================================================ �Լ� ���Ǻ� ================================================+
/**
 * @brief size ������ �迭 �����Ҵ� �� �ּ� ��ȯ
 * @param size �迭 ����
 */
float* makeArray(int size) {
    float* arr = (float*)malloc(sizeof(float) * size);
    return arr;
}

/**
 * @brief ���ڿ��� �յ� ������ ����
 * @param str ��� ���ڿ�
 */
void trim(char* str) {
    /*
        [��]
        ���ڿ��� ù ���ڰ� ���� ����(����, ��, �ٹٲ� ��)���� Ȯ���ϸ鼭 �����͸� ���� ���ڷ� �̵�.
        �� ������ ���� ���ڿ��� ���� ��ġ�� ������ �ƴ� ù ��ȿ ���ڷ� �̵���.
        [��]
        ���ڿ� ���������� isspace()�� ���� �������� �˻��ϸ鼭 �ڷ� �̵�.
        ������ �ƴ� ���ڸ� ������ �ű⼭ ����.
        [����]
        char str[] = "   \t Hello World!  \n";
        trim(str);
        printf("[%s]", str);   // -> ���: Hello World!
    */
    while (isspace((unsigned char)*str)) str++; // ���� ���� ����

    char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--; // ���� ���� ����

    *(end + 1) = '\0';
}

/**
 * @brief ��Ÿ������ ���Ͽ��� ��ü ���� ������ �о���� �Լ�
 *
 * @param img_width �̹��� �ʺ�
 * @param img_height �̹��� ����
 * @param output_nodes ��� ��� ����
 * @param num_hidden_layer hidden layer ����
 * @param hidden1_nodes ù ��° hidden layer ��� ��
 * @param hidden2_nodes �� ��° hidden layer ��� ��
 * @param hidden3_nodes �� ��° hidden layer ��� ��
 * @param rand_min �ʱ� ����ġ �ּҰ�
 * @param rand_max �ʱ� ����ġ �ִ밪
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
        printf("������ �������� �ʽ��ϴ� : %s\n", filepath);
        printf("[Warning]�н� ���� ���� �����Ͽ� ��Ÿ������ ������ �����ؾ� �մϴ�.\n");
        printf("[Warning]���丮 ������ Ȯ�����ּ���.\n");
        printf("[Warning]���� NUM_HIDDEN_LAYERS �� : %d\n", NUM_HIDDEN_LAYERS);
        exit(1);
    }

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        trim(line);

        // ��ũ�� ����� ���͸�: '# '�� �����ϴ� ��
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
                printf("[���� ����] %s\n", line);
            }
        }
    }
    printf("%s �κ��� �о�� ������\n", filepath);
    printf("IMG_WIDTH = %d\nIMG_HEIGHT = %d\n", *(img_width), *(img_height));
    printf("NUM_HIDDEN_LAYER = %d\n", *(num_hidden_layer));
    printf("HIDDEN1_NODES = %d\nHIDDEN2_NODES = %d\nHIDDEN3_NODES = %d\n", *(hidden1_nodes), *(hidden2_nodes), *(hidden3_nodes));
    printf("RAND_WEIGHT_MIN = %.2f\nRAND_WEIGHT_MAX = %.2f\n", *(rand_min), *(rand_max));

    //exit(0);
    fclose(fp);
}

/**
 * @brief ��Ÿ�����Ϳ��� Ư�� �±��� �迭 �����͸� �о���� �Լ�
 *
 * @param tag �±� �̸� (��: "WEIGHT1")
 * @param array ���� �����͸� ������ �迭
 * @param expected_size ���Ǵ� �迭 ũ�� (������)
 */
void loadArrayFromMetadata(const char* tag, float* array, int expected_size, const char* path) {
    char filepath[256];
    sprintf(filepath, "%s_h%d.txt", path, NUM_HIDDEN_LAYERS);
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL) {
        printf("������ �������� �ʽ��ϴ� : %s\n", filepath);
        printf("[Warning]�н� ���� ���� �����Ͽ� ��Ÿ������ ������ �����ؾ� �մϴ�.\n");
        printf("[Warning]���丮 ������ Ȯ�����ּ���.\n");
        printf("[Warning]���� NUM_HIDDEN_LAYERS �� : %d\n", NUM_HIDDEN_LAYERS);
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

    printf("[%s] : ��� %d��, ���� %d��\n", tag, expected_size, index);

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
        printf("�ùٸ��� ���� hidden layer ������ ���޵Ǿ����ϴ�. : %d\n", num_hidden_layer);
        break;
    }
}


/**
 * @brief �Ű���� forward propagation�� �����Ͽ� ��°��� ����ϴ� �Լ�
 *
 * @param input �Է� �迭
 * @param hidden1 ù ��° hidden layer �迭
 * @param hidden2 �� ��° hidden layer �迭
 * @param hidden3 �� ��° hidden layer �迭
 * @param output ��� �迭
 * @param weight1 input -> hidden1 ����ġ �迭
 * @param weight2 hidden1 -> hidden2 ����ġ �迭 // num_hidden_layer == 1 �̸� �갡 output����
 * @param weight3 hidden2 -> hidden3 ����ġ �迭 // num_hidden_layer == 2 �̸� �갡 output����
 * @param weight4 hidden3 -> output ����ġ �迭  // num_hidden_layer == 3 �̸� �갡 output����
 *
 * @param num_hidden_layer hidden layer ���� (1~3)
 * @param input_nodes �Է� ��� ��
 * @param output_nodes ��� ��� ��
 * @param hidden1_nodes ù ��° hidden layer ��� ��
 * @param hidden2_nodes �� ��° hidden layer ��� ��
 * @param hidden3_nodes �� ��° hidden layer ��� ��
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

    //printf("forwardPropagation() ����\n");

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
        printf("�ùٸ��� ���� hidden layer ������ ���޵Ǿ����ϴ�. : %d\n", num_hidden_layer);
    }
}

/**
 * @brief ��°� �� ���� ���� ���� ã�� ���� ����� ����ϴ� �Լ�
 *
 * @param output ��� ��� �迭
 * @param output_nodes ��� ��� ����
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

    printf("�ν� ���: %d (�����: %.3f)\n", recog, max_val);
}

/**
 * @brief �ñ׸��̵� Ȱ��ȭ �Լ�
 *
 * @param x �Է°�
 * @return float �ñ׸��̵� ���
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
        printf("�̹��� ���� ���� ����: %s\n", filename);
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
    char filepath[256];                     // ���ϸ� ����
    int correct = 0;                        // ���� ����
    int total = 0;                          // ��ü �õ� ����
    int correct_per_folder[10] = { 0 };     // ������ ���� ����
    int total_per_folder[10] = { 0 };       // ������ ��ü �õ� ����
    int num_files[10] = { 0 };              // ������ �Ϸ� �ε��� ���� �迭
    int folders_done;                       // �ݺ� ���� �÷���

    unsigned char* raw = (unsigned char*)malloc(sizeof(unsigned char) * IMG_WIDTH * IMG_HEIGHT);

    if (!raw) {
        printf("raw �迭 malloc ����\n");
        return;
    }
    for (int file_index = 0; ; file_index++) {
        folders_done = 0;

        for (int folder_index = 0; folder_index < OUTPUT_NODES; folder_index++) {
            // �ش� ������ �̹� �Ϸ�� ��� �ٷ� skip
            if (num_files[folder_index] != 0 && file_index >= num_files[folder_index]) {
                folders_done++;
                continue;
            }
            // ���� ��� ����
            sprintf(filepath, "%s%d/%d-%d.raw", filename, folder_index, folder_index, file_index);

            //printf("filepath:%s\n", filepath);
            FILE* fp = fopen(filepath, "rb");
            if (!fp) {
                printf("�ν��� ������ �� �̻� �������� ����: %s\n", filepath);
                if (num_files[folder_index] == 0) {
                    num_files[folder_index] = file_index;
                    total_per_folder[folder_index] = file_index;
                }
                folders_done++;
                continue;
            }

            fread(raw, sizeof(unsigned char), IMG_WIDTH * IMG_HEIGHT, fp);
            fclose(fp);

            // ����ȭ
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

            // ������ ã��
            int predicted = 0;
            float max_val = output[0];
            for (int i = 1; i < OUTPUT_NODES; i++) {
                if (output[i] > max_val) {
                    max_val = output[i];
                    predicted = i;
                }
            }
            // ��� ���
            printf("[�ν� ����: %s] -> ����: %d | ����: %d %s\n", filepath, predicted, folder_index, predicted == folder_index ? "O" : "X");

            if (predicted == folder_index) {
                correct++;
                correct_per_folder[folder_index]++;
            }

            total++;
        }
        // ��� ������ �Ϸ�Ǹ� ��� ����
        if (folders_done == OUTPUT_NODES) {
            printf("��� ���� �б� �Ϸ�\n");
            break;
        }
    }

    free(raw);
    float accuracy = (float)correct / total * 100.0f;

    printf("+---------------------------------------+\n");
    printf("��ü �νķ�: %.2f%% (%d / %d)\n", accuracy, correct, total);
    printf("+---------------------------------------+\n");
    printf("| ���� |  ��ü   |  ����   | �νķ�(%%)  |\n");
    printf("+------+---------+---------+------------+\n");

    for (int i = 0; i < OUTPUT_NODES; i++) {
        int total_count = total_per_folder[i];
        int correct_count = correct_per_folder[i];
        float acc = total_count > 0 ? (float)correct_count / total_count * 100.0f : 0.0f;

        printf("|  %2d  |  %5d  |  %5d  |   %6.2f%%  |\n", i, total_count, correct_count, acc);
    }
    printf("+---------------------------------------+\n");


}