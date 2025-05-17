#pragma warning(disable: 4996)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMG_WIDTH 512
#define IMG_HEIGHT 512
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT)

#define K 4
#define N 32
#define CLUSTER_FILE "../../output_txt/clustering_K%d_N%d.txt"
#define TEST_IMAGE_PATH "../../data/lena.img"
#define OUTPUT_IMAGE_PATH "../../img_reconst/lena_reconstructed_K%d_N%d.img"

float** load_centroids(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("중심 파일 열기 실패");
        return NULL;
    }

    float** X = (float**)malloc(sizeof(float*) * N);
    for (int i = 0; i < N; i++) {
        X[i] = (float*)malloc(sizeof(float) * K);
        for (int j = 0; j < K; j++) {
            fscanf(fp, "%f,", &X[i][j]);
        }
    }

    fclose(fp);
    return X;
}

unsigned char* read_image_file(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("테스트 이미지 열기 실패");
        return NULL;
    }

    unsigned char* buffer = (unsigned char*)malloc(sizeof(unsigned char) * IMG_SIZE);
    fread(buffer, sizeof(unsigned char), IMG_SIZE, fp);
    fclose(fp);
    return buffer;
}

float calculate_distance(float* a, float* b) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

int find_nearest_cluster(float* vec, float** X) {
    float min_dist = 1e9;
    int min_index = -1;
    for (int i = 0; i < N; i++) {
        float dist = calculate_distance(vec, X[i]);
        if (dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }
    return min_index;
}

int main() {
    // K와 N에 따라 클러스터 파일 경로를 동적으로 생성
    // 예: CLUSTER_FILE = "../../output_txt/clustering_K4_N32.txt"
    // 예: TEST_IMAGE_PATH = "../../data/lena.img"
    // 예: OUTPUT_IMAGE_PATH = "../../img_reconst/lena_reconstructed_K4_N32.img"
	// K와 N을 사용하여 클러스터 파일을 읽어온다

	char filename[256];
    sprintf(filename, CLUSTER_FILE, K, N);
    float** X = load_centroids(filename);

    unsigned char* image = read_image_file(TEST_IMAGE_PATH);

    unsigned char* reconstructed = (unsigned char*)malloc(sizeof(unsigned char) * IMG_SIZE);
    float* vec = (float*)malloc(sizeof(float) * K);

    int total_vectors = IMG_SIZE / K;

    double mse = 0.0;

    for (int i = 0; i < total_vectors; i++) {
        for (int j = 0; j < K; j++) {
            vec[j] = (float)image[i * K + j];
        }

        int nearest = find_nearest_cluster(vec, X);
        float dist = calculate_distance(vec, X[nearest]);
        mse += dist;


        for (int j = 0; j < K; j++) {
            int pixel_index = i * K + j;
            float value = X[nearest][j];
            if (value < 0) value = 0;
            if (value > 255) value = 255;
            reconstructed[pixel_index] = (unsigned char)(value + 0.5f);
        }
    }

	char output_image_path[256];
	sprintf(output_image_path, OUTPUT_IMAGE_PATH, K, N);
    FILE* out_fp = fopen(output_image_path, "wb");
    if (out_fp == NULL) {
        perror("재구성 이미지 저장 실패");
        return -1;
    }

    fwrite(reconstructed, sizeof(unsigned char), IMG_SIZE, out_fp);
    fclose(out_fp);

    printf("재구성 이미지 저장 완료: %s\n", output_image_path);
    mse /= total_vectors;
    printf("평가 이미지 MSE: %.4lf\n", mse);


    // 메모리 해제
    for (int i = 0; i < N; i++) free(X[i]);
    free(X);
    free(image);
    free(reconstructed);
    free(vec);

    return 0;
}
