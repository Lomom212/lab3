#include "gpu_reductions.h"
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void max_kernel(const float* image, std::uint32_t total_pixels, float* d_max_val) {
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_pixels) {
        float pixel = image[idx];
        atomicMax((int*)d_max_val, __float_as_int(pixel));
    }
}

double get_max_value(float* d_source_image, std::uint32_t total_pixels) {
    float h_max_result = 0.0f;
    float *d_max_result;

    cudaMalloc(&d_max_result, sizeof(float));
    cudaMemset(d_max_result, 0, sizeof(float));

    int threadsPerBlock = 256;
    int blocksperGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    max_kernel<<<blocksperGrid, threadsPerBlock>>>(d_source_image, total_pixels, d_max_result);

    cudaMemcpy(&h_max_result, d_max_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_max_result);

    float final_result = *reinterpret_cast<float*>(&h_max_result);
    return (double)final_result;
}