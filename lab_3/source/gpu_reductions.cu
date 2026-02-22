#include "gpu_reductions.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

#include "common.cuh"
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(__int_as_float(assumed), val)));
    } while (assumed != old);

    return __int_as_float(old);
}}

__global__ void max_kernel(const float* image, std::uint32_t total_pixels, float* d_max_val)
{
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_pixels) {
        atomicMaxDouble(d_max_val, image[idx]);
    }
}

double get_max_value(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width)
{
    float* d_image = static_cast<double*>(*d_source_image);
    std::uint32_t total_pixels = source_image_height * source_image_width;

    float* d_max_result;
    cudaMalloc(&d_max_result, sizeof(double));

    float init = -std::numeric_limits<double>::infinity();
    cudaMemcpy(d_max_result, &init, sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    max_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, total_pixels, d_max_result);

    float h_result;
    cudaMemcpy(&h_result, d_max_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_max_result);

    return static_cast<double>(h_result);
}