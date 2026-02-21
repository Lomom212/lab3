#include "gpu_matrix_convolution.h"
#include <cstdio>
#include <string>



__global__ void convolution_kernel(const float* img, uint32_t img_W, uint32_t img_H,
                                   const float* filter, uint32_t ker_W, uint32_t ker_H,
                                   float* res) {

    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_W && y < img_H) {
        float summe = 0.0f;
        int a = ker_W / 2;
        for (std::uint32_t i = 0; i < ker_H;i++) {
            for (std::uint32_t j = 0; j < ker_W; j++) {
                int current_Y = y - a + i;
                int current_X = x - a + j;
                float check = 0.0f;
                if (current_X >= 0 && current_X < img_W && current_Y < img_H && current_Y >= 0) {
                    check = img[current_Y * img_W + current_X];
                }
                summe += check * filter[i * ker_W + j];
            }
        }
        res[y * img_W + x] = summe;
    }



}


void matrix_convolution(void** d_source_matrix, std::uint32_t matrix_width, std::uint32_t matrix_height, void** d_kernel, std::uint32_t kernel_width, std::uint32_t kernel_height, void** d_result){

    size_t size = matrix_width * matrix_height * sizeof(float);
    float* d_res_ptr;
    cudaMalloc(&d_res_ptr, size);
    *d_result = (void*)d_res_ptr;

    dim3 threadsPerBlock(16,16);
    dim3 blocksperGrid((matrix_width+ threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (matrix_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    convolution_kernel<<<blocksperGrid, threadsPerBlock>>>((float*)*d_source_matrix, matrix_width, matrix_height,(float*)*d_kernel,kernel_width, kernel_height, d_res_ptr);

}