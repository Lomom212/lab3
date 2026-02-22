#include "gpu_utilities.h"
#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void sobel_math_combined_kernel(double* data_x, double* data_y, std::uint32_t size) {
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double gx = data_x[idx];
        double gy = data_y[idx];
        data_x[idx] = sqrt(gx * gx + gy * gy);
    }
}

void sobel_math_processing_combined(void* d_res_x, void* d_res_y, std::uint32_t num_pixels) {
    if (num_pixels == 0) return;
    int threads = 256;
    int blocks = (num_pixels + threads - 1) / threads;
    sobel_math_combined_kernel<<<blocks, threads>>>(static_cast<double*>(d_res_x), static_cast<double*>(d_res_y), num_pixels);
    cudaDeviceSynchronize();
}