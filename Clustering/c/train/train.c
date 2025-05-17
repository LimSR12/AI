#pragma warning(disable: 4996)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define IMG_WIDTH 512
#define IMG_HEIGHT 512
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT)

#define K 4
#define N 16
#define EPSILON 0.1f
#define MAX_ITER 100
#define THRESHOLD 0.0001f

#define NUM_TRAIN_IMAGES 19
#define TOTAL_VECTORS (IMG_SIZE * NUM_TRAIN_IMAGES / K)
#define CLUSTER_FILE "../../output_txt/clustering_K%d_N%d.txt"
#define FILE_PATH "../../data/I%d.img"

unsigned char* read_image_file(const char* filename);
void extract_vectors_from_image(unsigned char* image, float** v_array, int* index);
void initialize_centroids(float** v_array, float** x_array, int total_vectors);
void assign_clusters(float** v_array, float** x_array, int* cluster_assignments, int total_vectors);

float calculate_distance(float* a, float* b);

void update_centroids(float** v_array, float** x_array, int* cluster_assignments, int total_vectors);
void save_centroids_to_file(float** x_array, const char* filename);
void free_2d_float_array(float** array, int row_count);

int main() {
    // 1. 학습 벡터 메모리 할당
    float** v_array = (float**)malloc(sizeof(float*) * TOTAL_VECTORS);
    for (int i = 0; i < TOTAL_VECTORS; i++) {
        v_array[i] = (float*)malloc(sizeof(float) * K);
    }

    // 2. 이미지 읽고 벡터로 변환
    int current_vector_index = 0;
    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        char filename[256];
        sprintf(filename, FILE_PATH, i);
        printf("%s 읽어오기... ", filename);
        unsigned char* image = read_image_file(filename);
        extract_vectors_from_image(image, v_array, &current_vector_index);

        free(image);
        printf("완료\n");
    }

    // 3. 중심 벡터(X_i) 메모리 할당 및 초기화
    float** X = (float**)malloc(sizeof(float*) * N);
    for (int i = 0; i < N; i++) {
        X[i] = (float*)malloc(sizeof(float) * K);
    }
    
    initialize_centroids(v_array, X, TOTAL_VECTORS);
    
    // 4. 클러스터 할당
    int* cluster_assignments = (int*)malloc(sizeof(int) * TOTAL_VECTORS);
    assign_clusters(v_array, X, cluster_assignments, TOTAL_VECTORS);

    // 5. X 업데이트
    for (int iter = 0; iter < MAX_ITER; iter++) {
        assign_clusters(v_array, X, cluster_assignments, TOTAL_VECTORS);
        update_centroids(v_array, X, cluster_assignments, TOTAL_VECTORS);
        printf("%d번째 반복 완료\n", iter + 1);
    }

    // 6. X[i] 저장
    char save_filename[256];
    sprintf(save_filename, CLUSTER_FILE, K, N);
    save_centroids_to_file(X, save_filename);

    free_2d_float_array(v_array, TOTAL_VECTORS);
    free_2d_float_array(X, N);
    free(cluster_assignments);

	return 0;
}

unsigned char* read_image_file(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("파일 열기 실패");
        return NULL;
    }

    unsigned char* buffer = (unsigned char*)malloc(sizeof(unsigned char) * IMG_SIZE);
    if (buffer == NULL) {
        perror("메모리 할당 실패");
        fclose(fp);
        return NULL;
    }

    size_t read = fread(buffer, sizeof(unsigned char), IMG_SIZE, fp);
    if (read != IMG_SIZE) {
        fprintf(stderr, "파일 크기가 예상과 다릅니다. 읽은 픽셀 수: %zu\n", read);
        free(buffer);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return buffer;
}

void extract_vectors_from_image(unsigned char* image, float** v_array, int* index) {
    int num_vectors_in_image = IMG_SIZE / K;

    for (int i = 0; i < num_vectors_in_image; i++) {
        for (int j = 0; j < K; j++) {
            int pixel_index = i * K + j;
            v_array[*index][j] = (float)image[pixel_index];
        }
        (*index)++;
    }
}

void initialize_centroids(float** v_array, float** x_array, int total_vectors) {
    // 1. 모든 v_array 들의 평균 벡터를 계산해서 x_array에 저장
    for (int j = 0; j < K; j++) {
        float sum = 0.0f;
        for (int i = 0; i < total_vectors; i++) {
            sum += v_array[i][j];
        }
        x_array[0][j] = sum / total_vectors;
    }

    // 2. 중심을 epsilon 이용해서 분산
    int next = 1;
    while (next < N) {
        for (int dim = 0; dim < K; dim++) {
            if (next >= N) break;

            // +Epsilon
            for (int j = 0; j < K; j++) {
                x_array[next][j] = x_array[0][j];
            }
            x_array[next][dim] += EPSILON;
            next++;

            if (next >= N) break;

            // -Epsilon
            for (int j = 0; j < K; j++) {
                x_array[next][j] = x_array[0][j];
            }
            x_array[next][dim] -= EPSILON;
            next++;
        }
    }
    printf("중심 벡터 초기화 완료\n");
}

void assign_clusters(float** v_array, float** x_array, int* cluster_assignments, int total_vectors) {
    for (int i = 0; i < total_vectors; i++) {
        float min_dist = 1e9;
        int min_index = -1;

        for (int j = 0; j < N; j++) {
            float dist = calculate_distance(v_array[i], x_array[j]);
            if (dist < min_dist) {
                min_dist = dist;
                min_index = j;
            }
        }

        cluster_assignments[i] = min_index;
    }
}

float calculate_distance(float* a, float* b) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void update_centroids(float** v_array, float** x_array, int* cluster_assignments, int total_vectors) {
    // 초기화
    float** sum = (float**)malloc(sizeof(float*) * N);
    int* count = (int*)calloc(N, sizeof(int));

    for (int i = 0; i < N; i++) {
        sum[i] = (float*)calloc(K, sizeof(float));
    }

    // 1. 할당된 벡터들을 각 클러스터에 누적
    for (int i = 0; i < total_vectors; i++) {
        int cluster = cluster_assignments[i];
        count[cluster]++;
        for (int j = 0; j < K; j++) {
            sum[cluster][j] += v_array[i][j];
        }
    }

    // 2. 평균으로 나눠서 X 갱신
    for (int i = 0; i < N; i++) {
        if (count[i] == 0) continue; // 비어 있는 클러스터는 패스
        for (int j = 0; j < K; j++) {
            x_array[i][j] = sum[i][j] / count[i];
        }
    }

    for (int i = 0; i < N; i++) {
        free(sum[i]);
    }
    free(sum);
    free(count);
}

void save_centroids_to_file(float** x_array, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("중심 벡터 저장 파일 열기 실패");
        return;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            fprintf(fp, "%.6f", x_array[i][j]);
            if (j < K - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("중심 벡터 저장 완료: %s\n", filename);
}

void free_2d_float_array(float** array, int row_count) {
    for (int i = 0; i < row_count; i++) {
        free(array[i]);
    }
    free(array);
}
