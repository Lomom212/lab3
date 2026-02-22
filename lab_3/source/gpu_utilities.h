#pragma once
#include <cstdint>
#include "common.cuh"

void sobel_math_processing_combined(void* d_res_x, void* d_res_y, std::uint32_t num_pixels);