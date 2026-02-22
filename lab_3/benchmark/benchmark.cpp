#include "benchmark.h"
#include "cpu_reduction.h"
#include "gpu_reductions.h"

#include <cstdint>
#include <vector>
#include <iostream>

static void serial_benchmark(benchmark::State& state) {
	const auto n = state.range(0);
	uint32_t total_pixels = static_cast<uint32_t>(n) * n;

	std::vector<float> data(total_pixels, 1.0f);
	data[total_pixels - 1] = 500.0f;
	for (auto _ : state) {
		float result = get_max_value_serial(data.data(), total_pixels);
		benchmark::DoNotOptimize(result);
	}
}

BENCHMARK(serial_benchmark)->Unit(benchmark::kMillisecond)->Range(1024, 8192);

static void openmp_benchmark(benchmark::State& state) {
	const auto n = state.range(0);
	uint32_t total_pixels = static_cast<uint32_t>(n) * n;

	std::vector<float> data(total_pixels, 1.0f);
	data[total_pixels - 1] = 500.0f;
	for (auto _ : state) {
		float result = get_max_value_openmp(data.data(), total_pixels);
		benchmark::DoNotOptimize(result);
	}
}

BENCHMARK(openmp_benchmark)->Unit(benchmark::kMillisecond)->Range(1024, 8192);

static void cuda_benchmark(benchmark::State& state) {
	const auto n = state.range(0);
	uint32_t total_pixels = static_cast<uint32_t>(n) * n;

	std::vector<float> data(total_pixels, 1.0f);
	data[total_pixels - 1] = 500.0f;
	for (auto _ : state) {
		void* raw_ptr = static_cast<void*>(data.data());
		float result = get_max_value(&raw_ptr, 1, total_pixels);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(cuda_benchmark)->Unit(benchmark::kMillisecond)->Range(1024, 8192);

int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
