#include "gpu_LIN_upscaling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <iostream>

std::uint32_t get_LIN_upscaled_width(std::uint32_t image_width){
    if (image_width == 0) return 0;
    return (image_width * 2) -1;
}

std::uint32_t get_LIN_upscaled_height(std::uint32_t image_height){
    if (image_height == 0) return 0;
    return (image_height * 2) -1;
}

__global__ void lin_upscale_kernel(const double* source_image, double* result, std::uint32_t* source_width, std::uint32_t* result_width, std::uint32_t* result_height) {
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < result_width && y < result_height) {
        std::uint32_t inx = x/2;
        std::uint32_t iny = y/2;
        if (x%2 == 0 && y%2 == 0) {
            result[y*result_width + x] = source_image[iny*source_width + inx];
        }
        else if (y%2 != 0 && x%2 == 0) {
            double wert1 = source_image[iny*source_width + inx];
            double wert2 = source_image[iny*source_width + inx +1];
            result[y*result_width + x] = (wert1 + wert2) /2.0;
        }
        else if (x % 2 == 0 && y % 2 != 0) {
            double wert1 = source_image[iny*source_width + inx];
            double wert3 = source_image[(iny+1)*source_width + inx];
            result[y*result_width + x] = (wert1 + wert3) / 2.0;

        }
        else {
            double wert1 = source_image[iny*source_width + inx];
            double wert2 = source_image[iny*source_width + inx +1];
            double wert3 = source_image[(iny+1)*source_width + inx];
            double wert4 = source_image[(iny+1)*source_width + inx +1];
        }
    }
}

void LIN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    std::uint32_t width = get_LIN_upscaled_width(source_image_width);
    std::uint32_t height = get_LIN_upscaled_height(source_image_height);

    std::size_t size_ende = width*height * sizeof(double);
    cudaMalloc(d_result, size_ende);

    auto source_image = static_cast<const double *>(*d_source_image);
    double* result = static_cast<double *>(d_result);

    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x -1) / block_dim.x, (height+ block_dim.y -1) / block_dim.y);

    lin_upscale_kernel<<<grid_dim, block_dim>>>(source_image, result, source_image_width, source_image_height);

    cudaDeviceSynchronize();
}