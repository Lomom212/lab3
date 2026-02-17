cmake_minimum_required(VERSION 4.0)
project(lab_3_V1_4)

set(CMAKE_CXX_STANDARD 14)

include_directories(lab_3/benchmark)
include_directories(lab_3/source)
include_directories(lab_3/source/image)
include_directories(lab_3/source/io)
include_directories(lab_3/test)
include_directories(lab_3/test_compilation)

add_executable(lab_3_V1_4
        lab_3/benchmark/benchmark.cpp
        lab_3/benchmark/benchmark.h
        lab_3/source/image/bitmap_image.cpp
        lab_3/source/image/bitmap_image.h
        lab_3/source/image/pixel.h
        lab_3/source/io/image_parser.cpp
        lab_3/source/io/image_parser.h
        lab_3/source/authors.h
        lab_3/source/common.cuh
        lab_3/source/gpu_downsampling.cu
        lab_3/source/gpu_downsampling.h
        lab_3/source/gpu_LIN_upscaling.cu
        lab_3/source/gpu_LIN_upscaling.h
        lab_3/source/gpu_matrix_convolution.cu
        lab_3/source/gpu_matrix_convolution.h
        lab_3/source/gpu_memory_management.cu
        lab_3/source/gpu_memory_management.h
        lab_3/source/gpu_NN_upscaling.cu
        lab_3/source/gpu_NN_upscaling.h
        lab_3/source/gpu_reductions.cu
        lab_3/source/gpu_reductions.h
        lab_3/source/gpu_utilities.cu
        lab_3/source/gpu_utilities.h
        lab_3/source/grayscale_image.cpp
        lab_3/source/grayscale_image.h
        lab_3/source/intermediate_image.cpp
        lab_3/source/intermediate_image.cu
        lab_3/source/intermediate_image.h
        lab_3/source/main.cpp
        lab_3/source/utilities.cpp
        lab_3/source/utilities.h
        lab_3/test/lab_3_dev_cpu_grayscale.cpp
        lab_3/test/lab_3_dev_gpu_convolution.cu
        lab_3/test/lab_3_dev_gpu_NN_upscaling.cpp
        lab_3/test/lab_3_dev_gpu_reduction.cpp
        lab_3/test/lab_3_dev_gpu_up_and_download.cpp
        lab_3/test/test.cpp
        lab_3/test/test.h
        lab_3/test_compilation/test.cpp
        lab_3/test_compilation/test.h
        lab_3/test_compilation/test_compilation.cu)
