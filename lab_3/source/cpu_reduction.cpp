#include <cstdint>
#include "cpu_reduction.h"
#include <omp.h>
#include <limits>

double get_max_value_serial(const double* image, std::uint32_t size) {
    double max_val = -std::numeric_limits<double>::infinity();
    for (std::uint32_t i = 0; i < size; i++) {
        if (image[i] > max_val) {
            max_val = image[i];
        }
    }
    return max_val;
}

double get_max_value_openmp(const double* image, std::uint32_t size) {
    double max_value = -std::numeric_limits<double>::infinity();
#pragma omp parallel
    {
        double local_max = -std::numeric_limits<double>::infinity();
#pragma omp for
        for (std::uint32_t i = 0; i < size; i++) {
            if (image[i] > local_max) {
                local_max = image[i];
            }
        }
#pragma omp critical
        {
            if (local_max > max_value) {
                max_value = local_max;
            }
        }
    }
    return max_value;
}