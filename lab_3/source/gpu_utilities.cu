#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
#include "gpu_utilities.h"
#include <xmath.h>
#include <cstdint>

__global__ void sobel_kernel(float* f, std::uint32_t width, std::uint32_t height) {
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int id_x = y * width + x;
        float value = f[id_x];
        float f_strich = value * value;
        float f_double_strich = f_strich + f_strich;
        f[id_x] = sqrtf(f_double_strich);
    }
}

void gpu_sobel(float* d_matrix, std::uint32_t width, std::uint32_t height) {
    dim3 block(16,16);
    dim3 grid((width + block.x -1) / block.x, (height + block.y - 1) / block.y);

    sobel_kernel<<<grid, block>>>(d_matrix, width, height);
}


