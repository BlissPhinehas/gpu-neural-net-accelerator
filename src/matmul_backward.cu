#include "ops.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

// dA = dC * B^T
__global__ void matmul_backward_dA_kernel(float* B, float* dC, float* dA,
                                           int M, int K, int N) {
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    __shared__ float tiledC[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int dC_col = t * TILE_SIZE + threadIdx.x;
        tiledC[threadIdx.y][threadIdx.x] =
            (row < M && dC_col < N) ? dC[row * N + dC_col] : 0.0f;

        int B_row = t * TILE_SIZE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] =
            (col < K && B_row < N) ? B[col * N + B_row] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += tiledC[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < K)
        dA[row * K + col] = sum;
}

// dB = A^T * dC
// Straightforward: each thread computes one element of dB
// dB[k][j] = sum over i of A[i][k] * dC[i][j]
__global__ void matmul_backward_dB_kernel(float* A, float* dC, float* dB,
                                           int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tiledC[TILE_SIZE][TILE_SIZE];

    // This thread computes dB[row][col] where row=k, col=j
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // k dimension
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // j dimension

    float sum = 0.0f;
    int num_tiles = (M + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load A tile: we need A[i][row] = A[t*TILE+threadIdx.x][row]
        // tileA[ty][tx] = A[t*TILE+ty][row] — column slice of A
        int i_row = t * TILE_SIZE + threadIdx.y;
        int i_col = t * TILE_SIZE + threadIdx.x;

        // tileA[tx][ty] = A[t*TILE+tx][row] for dot with dC rows
        tileA[threadIdx.x][threadIdx.y] =
            (i_col < M && row < K) ? A[i_col * K + row] : 0.0f;

        // Load dC tile: dC[t*TILE+ty][col]
        tiledC[threadIdx.y][threadIdx.x] =
            (i_row < M && col < N) ? dC[i_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[k][threadIdx.y] * tiledC[k][threadIdx.x];

        __syncthreads();
    }

    if (row < K && col < N)
        dB[row * N + col] = sum;
}

void gpu_matmul_backward(float* A, float* B, float* dC,
                         float* dA, float* dB, int M, int K, int N) {
    float *d_A, *d_B, *d_dC, *d_dA, *d_dB;

    cudaMalloc(&d_A,  M * K * sizeof(float));
    cudaMalloc(&d_B,  K * N * sizeof(float));
    cudaMalloc(&d_dC, M * N * sizeof(float));
    cudaMalloc(&d_dA, M * K * sizeof(float));
    cudaMalloc(&d_dB, K * N * sizeof(float));

    cudaMemcpy(d_A,  A,  M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,  B,  K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dC, dC, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);

    dim3 blocks_dA((K + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_backward_dA_kernel<<<blocks_dA, threads>>>(d_B, d_dC, d_dA, M, K, N);

    dim3 blocks_dB((N + TILE_SIZE - 1) / TILE_SIZE,
                   (K + TILE_SIZE - 1) / TILE_SIZE);
    matmul_backward_dB_kernel<<<blocks_dB, threads>>>(d_A, d_dC, d_dB, M, K, N);

    cudaMemcpy(dA, d_dA, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dB, d_dB, K * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_dC);
    cudaFree(d_dA); cudaFree(d_dB);
}