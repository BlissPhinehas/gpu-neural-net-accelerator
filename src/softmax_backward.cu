#include "ops.h"
#include <cuda_runtime.h>

// Softmax backward uses the Jacobian-vector product formula:
// din[i] = softmax_out[i] * (dout[i] - dot(dout, softmax_out))
//
// Step 1: compute dot product of dout and softmax_out (a single number)
// Step 2: each thread computes its own din[i] using that dot product
//
// We compute the dot product using a parallel reduction (same idea as softmax forward)

__global__ void dot_product_kernel(float* a, float* b,
                                   float* block_dots, int size) {
    __shared__ float shared[256];

    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    shared[tid] = (i < size) ? a[i] * b[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    if (tid == 0) block_dots[blockIdx.x] = shared[0];
}

__global__ void softmax_backward_kernel(float* softmax_out, float* dout,
                                        float* din, float dot, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        din[i] = softmax_out[i] * (dout[i] - dot);
    }
}

void gpu_softmax_backward(float* softmax_out, float* dout,
                          float* din, int size) {
    float *d_sout, *d_dout, *d_din, *d_block_dots;

    cudaMalloc(&d_sout, size * sizeof(float));
    cudaMalloc(&d_dout, size * sizeof(float));
    cudaMalloc(&d_din,  size * sizeof(float));

    cudaMemcpy(d_sout, softmax_out, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout, dout,        size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (size + threads - 1) / threads;

    cudaMalloc(&d_block_dots, blocks * sizeof(float));

    // Step 1: parallel dot product
    dot_product_kernel<<<blocks, threads>>>(d_sout, d_dout, d_block_dots, size);

    float* h_block_dots = new float[blocks];
    cudaMemcpy(h_block_dots, d_block_dots, blocks * sizeof(float),
               cudaMemcpyDeviceToHost);

    float dot = 0.0f;
    for (int b = 0; b < blocks; b++) dot += h_block_dots[b];

    // Step 2: apply Jacobian-vector product
    softmax_backward_kernel<<<blocks, threads>>>(d_sout, d_dout, d_din, dot, size);

    cudaMemcpy(din, d_din, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_sout);
    cudaFree(d_dout);
    cudaFree(d_din);
    cudaFree(d_block_dots);
    delete[] h_block_dots;
}