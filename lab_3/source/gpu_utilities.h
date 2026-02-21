#pragma once
#include <cstdint>

#include "common.cuh"

void gpu_sobel(float* d_matrix, std::uint32_t width, std::uint32_t height);