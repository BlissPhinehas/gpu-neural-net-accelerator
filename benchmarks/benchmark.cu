#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "ops.h"

// Helper: fill an array with random floats between -1 and 1
void random_fill(float* arr, int size) {
    for (int i = 0; i < size; i++)
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

// Helper: time a GPU operation using CUDA events
// CUDA events are the correct way to time GPU code —
// they live on the GPU timeline, not the CPU clock
float time_gpu(void (*fn)(float*, float*, float*, int, int, int),
               float* A, float* B, float* C, int M, int K, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fn(A, B, C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // wait for GPU to finish

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// Helper: time a CPU operation using CUDA events for fair comparison
float time_cpu_matmul(float* A, float* B, float* C, int M, int K, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cpu_matmul(A, B, C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

void benchmark_matmul() {
    printf("\n=== Matrix Multiplication Benchmark ===\n");
    printf("%-10s %-12s %-12s %-10s\n", "Size", "CPU (ms)", "GPU (ms)", "Speedup");
    printf("------------------------------------------------\n");

    // Test across increasing matrix sizes
    int sizes[] = {128, 256, 512, 1024, 2048};
    int num_sizes = 5;

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s], K = sizes[s], N = sizes[s];

        float* A   = (float*)malloc(M * K * sizeof(float));
        float* B   = (float*)malloc(K * N * sizeof(float));
        float* C   = (float*)malloc(M * N * sizeof(float));

        random_fill(A, M * K);
        random_fill(B, K * N);

        float cpu_ms = time_cpu_matmul(A, B, C, M, K, N);
        float gpu_ms = time_gpu(gpu_matmul, A, B, C, M, K, N);
        float speedup = cpu_ms / gpu_ms;

        printf("%-10d %-12.2f %-12.2f %-10.1fx\n",
               sizes[s], cpu_ms, gpu_ms, speedup);

        free(A); free(B); free(C);
    }
}

void benchmark_relu() {
    printf("\n=== ReLU Benchmark ===\n");
    printf("%-12s %-12s %-12s %-10s\n", "Elements", "CPU (ms)", "GPU (ms)", "Speedup");
    printf("------------------------------------------------\n");

    int sizes[] = {1<<16, 1<<18, 1<<20, 1<<22, 1<<24};
    int num_sizes = 5;

    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        float* input  = (float*)malloc(size * sizeof(float));
        float* output = (float*)malloc(size * sizeof(float));
        random_fill(input, size);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cpu_relu(input, output, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cpu_ms = 0.0f;
        cudaEventElapsedTime(&cpu_ms, start, stop);

        cudaEventRecord(start);
        gpu_relu(input, output, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, start, stop);

        float speedup = cpu_ms / gpu_ms;
        printf("%-12d %-12.2f %-12.2f %-10.1fx\n",
               size, cpu_ms, gpu_ms, speedup);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(input); free(output);
    }
}

void benchmark_softmax() {
    printf("\n=== Softmax Benchmark ===\n");
    printf("%-12s %-12s %-12s %-10s\n", "Elements", "CPU (ms)", "GPU (ms)", "Speedup");
    printf("------------------------------------------------\n");

    int sizes[] = {1<<10, 1<<12, 1<<14, 1<<16, 1<<18};
    int num_sizes = 5;

    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        float* input  = (float*)malloc(size * sizeof(float));
        float* output = (float*)malloc(size * sizeof(float));
        random_fill(input, size);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cpu_softmax(input, output, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cpu_ms = 0.0f;
        cudaEventElapsedTime(&cpu_ms, start, stop);

        cudaEventRecord(start);
        gpu_softmax(input, output, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, start, stop);

        float speedup = cpu_ms / gpu_ms;
        printf("%-12d %-12.2f %-12.2f %-10.1fx\n",
               size, cpu_ms, gpu_ms, speedup);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(input); free(output);
    }
}

int main() {
    // Print GPU info so we know what hardware we're running on
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("CUDA Cores: %d\n", prop.multiProcessorCount * 128);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);

    benchmark_matmul();
    benchmark_relu();
    benchmark_softmax();

    printf("\nBenchmark complete.\n");
    return 0;
}