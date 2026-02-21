//
// Created by John on 21.02.2026.
//
#include <cstdint>
#include <omp.h>

float get_max_value_serial (const float* image, std::uint32_t size) {
    float max = 0.0f;
    for (std::uint32_t i = 0; i < size; i++) {
        if (image[i] > max) {
            max = image[i];
        }
    }
    return max;
}

//g++ -fopenmp source/cpu_reductions.cpp benchmark/benchmark.cpp -o benchmark_exe (in konsole beim kompilieren)

float get_max_value_openmp (const float* image, std::uint32_t size) {
    float max_value = 0.0f;
    #pragma omp parallel for reduction(max:max_value)
    for (std::uint32_t i = 0; i < size; i++) {
        if (image[i] > max_value) {
            max_value = image[i];
        }
    }
    return max_value;

}
