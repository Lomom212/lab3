#include "gpu_memory_management.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


void allocate_device_memory(IntermediateImage& image, void** devPtr){
    const std::size_t num_pixels = static_cast<std::size_t>(image.width) * image.height;
    const std::size_t size_in_bytes = num_pixels * sizeof(double);
    if (size_in_bytes > 0) {
        cudaMalloc(devPtr, size_in_bytes);
    } else {
        *devPtr = nullptr;
    }
}

void free_device_memory(void** devPtr){
    if (devPtr != nullptr && *devPtr != nullptr) {
        cudaFree(*devPtr);
        *devPtr = nullptr;
    }
}

void copy_data_to_device(IntermediateImage& image, void** devPtr){
    if (devPtr == nullptr || *devPtr == nullptr) {
        return;
    }

    const std::size_t size_in_bytes = image.pixels.size() * sizeof(double);

    if (size_in_bytes > 0) {

        cudaMemcpy(
            *devPtr,
            image.pixels.data(),
            size_in_bytes,
            cudaMemcpyHostToDevice
        );
    }
}

void copy_data_from_device(void** devPtr, IntermediateImage& image){
    if (devPtr == nullptr || *devPtr == nullptr) {
        return;
    }
    const std::size_t size_in_bytes = image.pixels.size() * sizeof(double);
    if (size_in_bytes > 0) {
        cudaMemcpy(image.pixels.data(), *devPtr, size_in_bytes, cudaMemcpyDeviceToHost);
    }
}