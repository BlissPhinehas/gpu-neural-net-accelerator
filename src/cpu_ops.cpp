#include "ops.h"
#include <math.h>
#include <float.h>

// ── Forward pass ──────────────────────────────────────────────────────────────

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

void cpu_relu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

void cpu_softmax(float* input, float* output, int size) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++)
        if (input[i] > max_val) max_val = input[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++)
        output[i] /= sum;
}

// ── Backward pass ─────────────────────────────────────────────────────────────

// matmul backward:
// dA = dC * B^T  (how much each element of A contributed to the loss)
// dB = A^T * dC  (how much each element of B contributed to the loss)
void cpu_matmul_backward(float* A, float* B, float* dC,
                         float* dA, float* dB, int M, int K, int N) {
    // dA = dC * B^T — shape (M x K)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += dC[i * N + j] * B[k * N + j];
            }
            dA[i * K + k] = sum;
        }
    }

    // dB = A^T * dC — shape (K x N)
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++) {
                sum += A[i * K + k] * dC[i * N + j];
            }
            dB[k * N + j] = sum;
        }
    }
}

// relu backward:
// gradient passes through where input > 0, blocked where input <= 0
void cpu_relu_backward(float* input, float* dout, float* din, int size) {
    for (int i = 0; i < size; i++) {
        din[i] = input[i] > 0.0f ? dout[i] : 0.0f;
    }
}

// softmax backward:
// din[i] = softmax_out[i] * (dout[i] - dot(dout, softmax_out))
// This is the Jacobian-vector product of softmax collapsed to one pass
void cpu_softmax_backward(float* softmax_out, float* dout,
                          float* din, int size) {
    // Compute dot product of dout and softmax_out
    float dot = 0.0f;
    for (int i = 0; i < size; i++)
        dot += dout[i] * softmax_out[i];

    // Apply Jacobian-vector product
    for (int i = 0; i < size; i++)
        din[i] = softmax_out[i] * (dout[i] - dot);
}