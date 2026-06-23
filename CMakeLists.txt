cmake_minimum_required(VERSION 3.18)
project(gpu_neural_net CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(src)

add_executable(benchmark
    benchmarks/benchmark.cu
    src/matmul.cu
    src/relu.cu
    src/softmax.cu
    src/matmul_backward.cu
    src/relu_backward.cu
    src/softmax_backward.cu
    src/cpu_ops.cpp
)

target_link_libraries(benchmark PRIVATE CUDA::cudart)

add_library(gpu_ops SHARED
    src/matmul.cu
    src/relu.cu
    src/softmax.cu
    src/matmul_backward.cu
    src/relu_backward.cu
    src/softmax_backward.cu
)

target_link_libraries(gpu_ops PRIVATE CUDA::cudart)