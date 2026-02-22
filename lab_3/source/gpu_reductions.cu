#include "gpu_reductions.h"
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void max_kernel(const float* image, std::uint32_t width, std::uint32_t height, float* d_max_val) {
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float pixel = image[y * width + x];
        atomicMax((int*)d_max_val, __float_as_int(pixel));
    }
}

double get_max_value(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width){
    float h_max_result = 0.0f;
    float *d_max_result;
    float *d_img = (float*)(*d_source_image);

    cudaMalloc(&d_max_result, sizeof(float));
    cudaMemset(d_max_result,0,sizeof(float));

    dim3 threadsPerBlock(16,16);
    dim3 blocksperGrid((source_image_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (source_image_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    max_kernel<<<blocksperGrid, threadsPerBlock>>>(d_img, source_image_width, source_image_height, d_max_result);
    cudaMemcpy(&h_max_result,d_max_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_max_result);

    float final_result = __int_as_float(*(int*)&h_max_result);
    return (double)final_result;
}