#include <cstdint>
#include <omp.h>
#include <limits>
#include <vector>

double get_max_value_serial(const double* image, std::uint32_t size) {
    double max_val = -std::numeric_limits<double>::infinity(); // Sicherer Startwert
    for (std::uint32_t i = 0; i < size; i++) {
        if (image[i] > max_val) {
            max_val = image[i];
        }
    }
    return max_val;
}

double get_max_value_openmp(const double* image, std::uint32_t size) {
    double max_value = -std::numeric_limits<double>::infinity();
    // OpenMP Max-Reduktion ist ab Version 3.1 Standard
#pragma omp parallel for reduction(max:max_value)
    for (std::uint32_t i = 0; i < size; i++) {
        if (image[i] > max_value) {
            max_value = image[i];
        }
    }
    return max_value;
}
