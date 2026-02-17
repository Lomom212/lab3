#include "gpu_NN_upscaling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <iostream>

__global__ void nn_upscale_kernel(double* source, double dist, uint32_t )

std::uint32_t get_NN_upscaled_width(std::uint32_t image_width){
    return image_width*3;
}

std::uint32_t get_NN_upscaled_height(std::uint32_t image_height){
    return image_height*3;
}


void NN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    uint32_t res_width = get_NN_upscaled_width(source_image_width);
    uint32_t res_height = get_NN_upscaled_height(source_image_height);
    size_t res_size = res_width*res_height * sizeof(double);

    cudaError_t error = cudaMalloc(d_result, res_size);

}