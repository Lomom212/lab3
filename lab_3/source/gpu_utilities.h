#pragma once
#include <cstdint>
#include "common.cuh"

void sobel_math_processing(void* d_result, std::uint32_t num_pixels);