#include "gpu_downsampling.h"
#include "gpu_memory_management.h"
#include <cstdio>

std::uint32_t get_downsampled_width(std::uint32_t image_width){
    if (image_width == 0) return 0;
    return (image_width / 3) +1;
}

std::uint32_t get_downsampled_height(std::uint32_t image_height){
    if (image_height == 0) return 0;
    return (image_height / 3) +1;
}

__global__ void downsample_kernel(const double* d_source, double* d_result, std::uint32_t widthsource, std::uint32_t heightsource, std::uint32_t result_width, std::uint32_t result_height) {
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < result_width && y < result_height) {
        double s = 0.0;
        double p = 0.0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int inx = (int) (x*3) +dx;
                int iny = (int) (y*3) +dy;

                if (inx >= 0 && inx < (int) widthsource && iny >= 0 && iny < (int) heightsource) {
                    s += d_source[iny * widthsource + inx];
                    p+= 1.0;
                }
            }
        }
        if (p > 0.0) {
            d_result[y * result_width + x] = s /p;
        }
    }

}

void image_downsampling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    std::uint32_t widthsource = get_downsampled_width(source_image_width);
    std::uint32_t heightsource = get_downsampled_height(source_image_height);

    if (widthsource == 0 || heightsource == 0) return;
    std::size_t size = widthsource * heightsource * sizeof(double);
    cudaMalloc(d_result, size);

    auto d_source = static_cast<const double *>(*d_source_image);
    double* dresult = static_cast<double *>(*d_result);

    dim3 blockDim(16, 16);
    dim3 gridDim((widthsource + blockDim.x -1) / blockDim.x, (heightsource + blockDim.y -1) / blockDim.y);

    downsample_kernel<<<gridDim, blockDim>>>(d_source, dresult, source_image_width, source_image_height, widthsource, heightsource);

    cudaDeviceSynchronize();
}