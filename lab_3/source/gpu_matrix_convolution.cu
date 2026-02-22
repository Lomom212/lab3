#include "gpu_matrix_convolution.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void convolution_kernel(const double* img, uint32_t img_W, uint32_t img_H,
                                   const double* filter, uint32_t ker_dim,
                                   double* res) {

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_W && y < img_H) {
        double summe = 0.0;
        int a = ker_dim / 2;

        for (uint32_t i = 0; i < ker_dim; ++i) {
            for (uint32_t j = 0; j < ker_dim; ++j) {

                // i ist der Zeilenindex (y), j ist der Spaltenindex (x) des Kernels
                int src_x = static_cast<int>(x) - static_cast<int>(j) + a;
                int src_y = static_cast<int>(y) - static_cast<int>(i) + a;

                if (src_x >= 0 && src_x < static_cast<int>(img_W) &&
                    src_y >= 0 && src_y < static_cast<int>(img_H)) {
                    summe += img[src_y * img_W + src_x] * filter[i * ker_dim + j];
                }
            }
        }
        res[y * img_W + x] = summe;
    }
}

void matrix_convolution(void** d_source_matrix, std::uint32_t matrix_width, std::uint32_t matrix_height,
                        void** d_kernel, std::uint32_t kernel_width, std::uint32_t kernel_height,
                        void** d_result) {

    size_t size = static_cast<size_t>(matrix_width) * matrix_height * sizeof(double);
    double* d_res_ptr = nullptr;

    cudaMalloc(&d_res_ptr, size);
    *d_result = reinterpret_cast<void*>(d_res_ptr);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksperGrid((matrix_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (matrix_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolution_kernel<<<blocksperGrid, threadsPerBlock>>>(
        static_cast<const double*>(*d_source_matrix),
        matrix_width,
        matrix_height,
        static_cast<const double*>(*d_kernel),
        kernel_width,
        d_res_ptr
    );
}