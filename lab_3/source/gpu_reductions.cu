#include "gpu_reductions.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

#include "common.cuh"

__device__ double atomicMaxDouble(double* address, double val)
{
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull, assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double max_val = fmax(old_val, val);
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(max_val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void max_kernel(const double* image, std::uint32_t total_pixels, double* d_max_val)
{
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_pixels) {
        atomicMaxDouble(d_max_val, image[idx]);
    }
}

double get_max_value(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width)
{
    double* d_image = static_cast<double*>(*d_source_image);
    std::uint32_t total_pixels = source_image_height * source_image_width;

    double* d_max_result;
    cudaMalloc(&d_max_result, sizeof(double));

    double init = -std::numeric_limits<double>::infinity();
    cudaMemcpy(d_max_result, &init, sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    max_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, total_pixels, d_max_result);

    double h_result;
    cudaMemcpy(&h_result, d_max_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_max_result);

    return h_result;
}