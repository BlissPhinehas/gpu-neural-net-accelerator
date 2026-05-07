#include "ops.h"
#include <math.h>
#include <float.h>

// Naive CPU matrix multiplication
// For every output cell C[i][j], we dot-product row i of A with column j of B
void cpu_matmul(float* A, float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU ReLU: clamp every element to 0 if negative
void cpu_relu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

// CPU softmax: numerically stable version
// Step 1: find max value (prevents overflow when exponentiating)
// Step 2: subtract max and exponentiate
// Step 3: divide by sum to normalize
void cpu_softmax(float* input, float* output, int size) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}