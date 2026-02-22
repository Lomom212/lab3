#ifndef CPU_REDUCTIONS_H
#define CPU_REDUCTIONS_H

#include <cstdint>

/**
 * @brief Serielle Referenzlösung zur Bestimmung des Maximalwerts (Aufgabe 8a).
 * * @param image Pointer auf die Bilddaten (Host-Speicher).
 * @param size Gesamte Anzahl der Pixel.
 * @return Maximalwert als double.
 */
double get_max_value_serial(const double* image, std::uint32_t size);

/**
 * @brief OpenMP-parallelisierte Referenzlösung zur Bestimmung des Maximalwerts (Aufgabe 8a).
 * * @param image Pointer auf die Bilddaten (Host-Speicher).
 * @param size Gesamte Anzahl der Pixel.
 * @return Maximalwert als double.
 */
double get_max_value_openmp(const double* image, std::uint32_t size);

#endif // CPU_REDUCTIONS_H

