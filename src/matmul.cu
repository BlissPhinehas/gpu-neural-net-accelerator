#include "ops.h"
#include <cuda_runtime.h>

// TILE_SIZE: how many elements wide/tall each tile is
// We load 16x16 chunks of A and B into shared memory at a time
// Shared memory is ~100x faster than global GPU memory
#define TILE_SIZE 16

__global__ void matmul_kernel(float* A, float* B, float* C,
                               int M, int K, int N) {
    // Each block computes one TILE_SIZE x TILE_SIZE chunk of output C
    // Shared memory tiles — live in fast on-chip memory
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Which output cell this thread is responsible for
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles along the K dimension
    // Each iteration loads one tile of A and one tile of B
    // then all threads in the block multiply their elements together
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {

        // Collaboratively load tile of A into shared memory
        // Each thread loads one element
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Collaboratively load tile of B into shared memory
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Wait for ALL threads to finish loading before computing
        // Without this, some threads might read tiles before they're written
        __syncthreads();

        // Each thread computes its dot product contribution from this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Wait for ALL threads to finish computing before loading next tile
        __syncthreads();
    }

    // Write result to output matrix
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void gpu_matmul(float* A, float* B, float* C, int M, int K, int N) {
    float *d_A, *d_B, *d_C;

    // Allocate GPU memory for all three matrices
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy A and B from CPU to GPU
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch config: 2D grid of 2D blocks
    // Each block = 16x16 threads = 256 threads
    // Grid covers the entire output matrix C
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

    // Copy result back to CPU
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}