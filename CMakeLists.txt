cmake_minimum_required(VERSION 3.18)
project(gpu_neural_net CUDA CXX)

# Require CUDA
find_package(CUDAToolkit REQUIRED)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# GPU architecture — T4 is sm_75
set(CMAKE_CUDA_ARCHITECTURES 75)

# Include headers from src/
include_directories(src)

# Benchmarks executable
add_executable(benchmark
    benchmarks/benchmark.cu
    src/matmul.cu
    src/relu.cu
    src/softmax.cu
    src/cpu_ops.cpp
)

target_link_libraries(benchmark PRIVATE CUDA::cudart)

# Shared library for Python ctypes wrapper
add_library(gpu_ops SHARED
    src/matmul.cu
    src/relu.cu
    src/softmax.cu
)

target_link_libraries(gpu_ops PRIVATE CUDA::cudart)
