#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward pass
void gpu_matmul(float* A, float* B, float* C, int M, int K, int N);
void gpu_relu(float* input, float* output, int size);
void gpu_softmax(float* input, float* output, int size);

// Backward pass
void gpu_matmul_backward(float* A, float* B, float* dC,
                         float* dA, float* dB, int M, int K, int N);
void gpu_relu_backward(float* input, float* dout,
                       float* din, int size);
void gpu_softmax_backward(float* softmax_out, float* dout,
                          float* din, int size);

// CPU equivalents
void cpu_matmul(float* A, float* B, float* C, int M, int K, int N);
void cpu_relu(float* input, float* output, int size);
void cpu_softmax(float* input, float* output, int size);
void cpu_matmul_backward(float* A, float* B, float* dC,
                         float* dA, float* dB, int M, int K, int N);
void cpu_relu_backward(float* input, float* dout,
                       float* din, int size);
void cpu_softmax_backward(float* softmax_out, float* dout,
                          float* din, int size);

#ifdef __cplusplus
}
#endif