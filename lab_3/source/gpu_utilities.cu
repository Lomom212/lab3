#include "gpu_utilities.h"

#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void sobel_math_kernel(double* data, std::uint32_t size) {
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double f = data[idx];
        data[idx] = sqrt(f * f + f * f);
    }
}

void sobel_math_processing(void* d_result, std::uint32_t num_pixels) {
    if (num_pixels == 0) return;

    int threads = 256;
    int blocks = (num_pixels + threads - 1) / threads;

    sobel_math_kernel<<<blocks, threads>>>(static_cast<double*>(d_result), num_pixels);
    cudaDeviceSynchronize();
}

