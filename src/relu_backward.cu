#include "ops.h"
#include <cuda_runtime.h>

// ReLU backward is the simplest of the three:
// The gradient passes through unchanged where the original input was > 0
// and is blocked (set to 0) where the original input was <= 0
// This is called the "subgradient" — ReLU has no gradient at exactly 0
__global__ void relu_backward_kernel(float* input, float* dout,
                                     float* din, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        din[i] = input[i] > 0.0f ? dout[i] : 0.0f;
    }
}

void gpu_relu_backward(float* input, float* dout, float* din, int size) {
    float *d_input, *d_dout, *d_din;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_dout,  size * sizeof(float));
    cudaMalloc(&d_din,   size * sizeof(float));

    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout,  dout,  size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(d_input, d_dout, d_din, size);

    cudaMemcpy(din, d_din, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_dout);
    cudaFree(d_din);
}