#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Matrix multiplication: C = A * B
// A is (M x K), B is (K x N), C is (M x N)
void gpu_matmul(float* A, float* B, float* C, int M, int K, int N);

// ReLU activation: out[i] = max(0, in[i])
void gpu_relu(float* input, float* output, int size);

// Softmax: numerically stable parallel softmax over a vector
void gpu_softmax(float* input, float* output, int size);

// CPU equivalents for benchmarking
void cpu_matmul(float* A, float* B, float* C, int M, int K, int N);
void cpu_relu(float* input, float* output, int size);
void cpu_softmax(float* input, float* output, int size);

#ifdef __cplusplus
}
#endif