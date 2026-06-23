#include "ops.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

// dA = dC * B^T
// Each thread computes one element of dA
// Row i, col k of dA = dot product of row i of dC with col k of B^T
//                    = dot product of row i of dC with row k of B
__global__ void matmul_backward_dA_kernel(float* B, float* dC, float* dA,
                                           int M, int K, int N) {
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    __shared__ float tiledC[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // indexes M
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // indexes K

    float sum = 0.0f;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile of dC: row stays same, col comes from tile
        int dC_col = t * TILE_SIZE + threadIdx.x;
        tiledC[threadIdx.y][threadIdx.x] =
            (row < M && dC_col < N) ? dC[row * N + dC_col] : 0.0f;

        // Load tile of B (we need B^T so we swap indices when loading)
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
// Each thread computes one element of dB
// Row k, col j of dB = dot product of col k of A with col j of dC
//                    = dot product of row k of A^T with col j of dC
__global__ void matmul_backward_dB_kernel(float* A, float* dC, float* dB,
                                           int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tiledC[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // indexes K
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // indexes N

    float sum = 0.0f;
    int num_tiles = (M + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile of A^T (swap indices when loading)
        int A_row = t * TILE_SIZE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] =
            (row < K && A_row < M) ? A[A_row * K + row] : 0.0f;

        // Load tile of dC
        int dC_row = t * TILE_SIZE + threadIdx.y;
        tiledC[threadIdx.y][threadIdx.x] =
            (dC_row < M && col < N) ? dC[dC_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tiledC[k][threadIdx.x];

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

    // Launch dA kernel — output is M x K
    dim3 blocks_dA((K + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_backward_dA_kernel<<<blocks_dA, threads>>>(d_B, d_dC, d_dA, M, K, N);

    // Launch dB kernel — output is K x N
    dim3 blocks_dB((N + TILE_SIZE - 1) / TILE_SIZE,
                   (K + TILE_SIZE - 1) / TILE_SIZE);
    matmul_backward_dB_kernel<<<blocks_dB, threads>>>(d_A, d_dC, d_dB, M, K, N);

    cudaMemcpy(dA, d_dA, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dB, d_dB, K * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_dC);
    cudaFree(d_dA); cudaFree(d_dB);
}