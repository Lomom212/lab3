#include <cuda_runtime_api.h>
#include "intermediate_image.h"
#include "gpu_matrix_convolution.h"
#include "gpu_utilities.h"

void IntermediateImage::apply_sobel_filter() {
    if (pixels.empty()) return;

    std::uint32_t num_pixels = width * height;
    size_t img_bytes = num_pixels * sizeof(double);

    double h_kernel_x[9] = {
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    };

    double h_kernel_y[9] = {
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
    };

    void* d_img = nullptr;
    void* d_kernel_x = nullptr;
    void* d_kernel_y = nullptr;
    void* d_res_x = nullptr;
    void* d_res_y = nullptr;

    cudaMalloc(&d_img, img_bytes);
    cudaMalloc(&d_kernel_x, 9 * sizeof(double));
    cudaMalloc(&d_kernel_y, 9 * sizeof(double));

    cudaMemcpy(d_img, pixels.data(), img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_x, h_kernel_x, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_y, h_kernel_y, 9 * sizeof(double), cudaMemcpyHostToDevice);

    matrix_convolution(&d_img, width, height, &d_kernel_x, 3, 3, &d_res_x);
    matrix_convolution(&d_img, width, height, &d_kernel_y, 3, 3, &d_res_y);

    sobel_math_processing_combined(d_res_x, d_res_y, num_pixels);

    cudaMemcpy(pixels.data(), d_res_x, img_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_kernel_x);
    cudaFree(d_kernel_y);
    cudaFree(d_res_x);
    cudaFree(d_res_y);
}