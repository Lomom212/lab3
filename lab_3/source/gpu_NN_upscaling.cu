#include "gpu_NN_upscaling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <iostream>


std::uint32_t get_NN_upscaled_width(std::uint32_t image_width){
    return image_width*3;
}

std::uint32_t get_NN_upscaled_height(std::uint32_t image_height){
    return image_height*3;
}

__global__ void nn_upscale_kernel(const double* source, double* resul, double source_width, std::uint32_t width, std::uint32_t height) {
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        std::uint32_t in_x = x / 3;
        std::uint32_t in_y = y / 3;


        resul[y * width + x] = source[in_y * source_width + in_x];
    }
}

void NN_image_upscaling(void* d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    std::uint32_t width = get_NN_upscaled_width(source_image_width);
    std::uint32_t height = get_NN_upscaled_height(source_image_height);
    std::size_t size_ende = (std::size_t) width*height * sizeof(double);

    cudaMalloc(d_result, size_ende);

    const double *source = reinterpret_cast<const double *>(d_source_image);

    double* result = reinterpret_cast<double*>(*d_result);

    dim3 block_dim(16, 16);
    dim3 grid_dim((width+ block_dim.x-1) / block_dim.x, (height + block_dim.y-1) / block_dim.y);

    nn_upscale_kernel<<<grid_dim, block_dim>>>(source, result, source_image_width, width, height);
    cudaDeviceSynchronize();
}