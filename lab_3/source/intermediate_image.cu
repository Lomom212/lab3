#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
#include "intermediate_image.h"

#include <complex>

#include "gpu_matrix_convolution.h"
#include "gpu_utilities.h"

void IntermediateImage::apply_sobel_filter(){

    std::uint32_t s_width = this->width;
    std::uint32_t s_height = this->height;
    size_t image_size = s_width * s_height * sizeof(float);

    float h_kernel[9] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f
    };
    std::uint32_t K_width = 3;
    std::uint32_t K_height = 3;

    float *d_img, *d_kernel, *d_result;
    cudaMalloc(&d_img, image_size);
    cudaMalloc(&d_kernel, 9 * sizeof(float));
    cudaMemcpy(d_img, this->pixels, image_size, cudaMemcpyHostToDevice);

    matrix_convolution((void**)&d_img, width, height, (void**)&d_kernel, 3, 3, (void**)&d_result);

    gpu_sobel(d_result, width, height);


    cudaMemcpy(this->pixels, d_result, image_size, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_kernel);
    cudaFree(d_result);
}