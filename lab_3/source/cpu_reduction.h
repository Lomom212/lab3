//
// Created by John on 21.02.2026.
//

#ifndef LAB_3_V1_4_CPU_REDUCTION_H
#define LAB_3_V1_4_CPU_REDUCTION_H
#include <cstdint>

#endif //LAB_3_V1_4_CPU_REDUCTION_H

float get_max_value_serial(const float* image, std::uint32_t size);
float get_max_value_openmp(const float* image, std::uint32_t size);