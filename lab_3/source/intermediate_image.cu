#include <cuda_runtime_api.h>
#include "intermediate_image.h"
#include "gpu_matrix_convolution.h"
#include "gpu_utilities.h"

void IntermediateImage::apply_sobel_filter() {
    if (pixels.empty()) return;

    std::uint32_t num_pixels = width * height;
    size_t img_bytes = num_pixels * sizeof(double);

    double h_kernel[9] = {
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    };

    void* d_img = nullptr;
    void* d_kernel = nullptr;
    void* d_result = nullptr;

    cudaMalloc(&d_img, img_bytes);
    cudaMalloc(&d_kernel, 9 * sizeof(double));

    cudaMemcpy(d_img, pixels.data(), img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(double), cudaMemcpyHostToDevice);

    matrix_convolution(&d_img, width, height, &d_kernel, 3, 3, &d_result);

    sobel_math_processing(d_result, num_pixels);

    cudaMemcpy(pixels.data(), d_result, img_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_kernel);
    cudaFree(d_result);
}