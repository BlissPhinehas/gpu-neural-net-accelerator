#include "ops.h"
#include <cuda_runtime.h>
#include <float.h>

// Step 1 kernel: find the maximum value in the array
// Each thread checks one element, then threads in the same block
// compare their results using shared memory (fast on-chip memory)
// __shared__ means this memory is shared between all threads in a block
__global__ void max_kernel(float* input, float* block_maxes, int size) {
    __shared__ float shared[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread loads one element, or -infinity if out of bounds
    shared[tid] = (i < size) ? input[i] : -FLT_MAX;
    __syncthreads(); // wait for all threads in block to finish loading

    // Reduction: threads compare pairs of values, halving active
    // threads each round until one value remains per block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (shared[tid + stride] > shared[tid])
                shared[tid] = shared[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes its block's max to global memory
    if (tid == 0) block_maxes[blockIdx.x] = shared[0];
}

// Step 2 kernel: subtract max, exponentiate, accumulate sum per block
__global__ void exp_sum_kernel(float* input, float* output,
                                float* block_sums, float max_val, int size) {
    __shared__ float shared[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = 0.0f;
    if (i < size) {
        val = expf(input[i] - max_val);
        output[i] = val;
    }
    shared[tid] = val;
    __syncthreads();

    // Reduction to sum all values in this block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }

    if (tid == 0) block_sums[blockIdx.x] = shared[0];
}

// Step 3 kernel: divide every element by the total sum
__global__ void normalize_kernel(float* output, float total_sum, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) output[i] /= total_sum;
}

void gpu_softmax(float* input, float* output, int size) {
    float *d_input, *d_output;
    cudaMalloc(&d_input,  size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (size + threads - 1) / threads;

    // Allocate space for per-block results
    float *d_block_maxes, *d_block_sums;
    cudaMalloc(&d_block_maxes, blocks * sizeof(float));
    cudaMalloc(&d_block_sums,  blocks * sizeof(float));

    // Step 1: find max across all blocks
    max_kernel<<<blocks, threads>>>(d_input, d_block_maxes, size);

    // Copy block maxes to CPU and find the global max
    float* h_block_maxes = new float[blocks];
    cudaMemcpy(h_block_maxes, d_block_maxes, blocks * sizeof(float),
               cudaMemcpyDeviceToHost);
    float global_max = -FLT_MAX;
    for (int b = 0; b < blocks; b++)
        if (h_block_maxes[b] > global_max) global_max = h_block_maxes[b];

    // Step 2: subtract max, exponentiate, get per-block sums
    exp_sum_kernel<<<blocks, threads>>>(d_input, d_output,
                                        d_block_sums, global_max, size);

    // Copy block sums to CPU and find total sum
    float* h_block_sums = new float[blocks];
    cudaMemcpy(h_block_sums, d_block_sums, blocks * sizeof(float),
               cudaMemcpyDeviceToHost);
    float total_sum = 0.0f;
    for (int b = 0; b < blocks; b++) total_sum += h_block_sums[b];

    // Step 3: normalize
    normalize_kernel<<<blocks, threads>>>(d_output, total_sum, size);

    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_maxes);
    cudaFree(d_block_sums);
    delete[] h_block_maxes;
    delete[] h_block_sums;
}