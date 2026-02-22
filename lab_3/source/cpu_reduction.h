#ifndef CPU_REDUCTIONS_H
#define CPU_REDUCTIONS_H

#include <cstdint>

double get_max_value_serial(const double* image, std::uint32_t size);
double get_max_value_openmp(const double* image, std::uint32_t size);

#endif