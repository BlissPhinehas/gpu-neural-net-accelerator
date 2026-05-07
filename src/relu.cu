#include "ops.h"
#include <cuda_runtime.h>

// This is the GPU kernel function — the __global__ keyword means
// "this runs on the GPU, called from the CPU"
// Every GPU thread runs this function simultaneously on one element
// threadIdx.x = thread's position inside its block
// blockIdx.x  = which block this thread belongs to
// blockDim.x  = how many threads are in each block
// Together they give each thread a unique index into the array
__global__ void relu_kernel(float* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: don't go past the end of the array
    // (total threads launched may be slightly more than size)
    if (i < size) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

void gpu_relu(float* input, float* output, int size) {
    float *d_input, *d_output;

    // Allocate memory on the GPU (d_ prefix = "device" = GPU)
    cudaMalloc(&d_input,  size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data from CPU memory to GPU memory
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    // 256 threads per block is a standard starting point
    // We calculate how many blocks we need to cover the whole array
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<blocks, threads_per_block>>>(d_input, d_output, size);

    // Copy results back from GPU memory to CPU memory
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
}