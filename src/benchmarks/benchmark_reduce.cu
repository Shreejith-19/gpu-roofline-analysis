#include "reduce.cuh"
#include "cuda_utils.cuh"
#include "common.hpp"

#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

struct ReduceBenchmarkConfig {
    std::string name;
    size_t N;
    int iterations;
};

void runReduceBenchmark(const ReduceBenchmarkConfig& cfg, gpu_roofline::CsvWriter* csvWriter) {
    size_t bytes = cfg.N * sizeof(float);
    std::vector<float> h_input(cfg.N);
    for (size_t i = 0; i < cfg.N; ++i) {
        h_input[i] = 1.0f; // Uniform 1.0f: theoretical sum should be exactly N
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // Allocate persistent scratchpad buffer for benchmarking to isolate pure kernel execution time
    size_t tempBytes = gpu_roofline::getReduceSumTempStorageBytes(cfg.N, 256);
    float* d_temp = nullptr;
    if (tempBytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_temp, tempBytes));
    }

    // Warm-up launch
    gpu_roofline::launchReduceSum(d_input, d_output, cfg.N, d_temp);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify computed result once
    float h_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    float expectedSum = static_cast<float>(cfg.N);
    float diff = std::fabs(h_result - expectedSum);
    if (diff > 1e-3f * expectedSum && cfg.N < 16777216) {
        std::cerr << "[WARN] Reduction result verification drift: got " << h_result 
                  << ", expected " << expectedSum << std::endl;
    }

    // Timed execution
    gpu_roofline::GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < cfg.iterations; ++iter) {
        gpu_roofline::launchReduceSum(d_input, d_output, cfg.N, d_temp);
    }
    timer.stop();
    float totalMs = timer.elapsedMillis();
    float avgMs = totalMs / cfg.iterations;

    // Metrics calculations
    double seconds = avgMs / 1000.0;
    double bytesRead = static_cast<double>(bytes);
    double effectiveGBs = (bytesRead / 1e9) / seconds;
    double gElementsPerSec = (static_cast<double>(cfg.N) / 1e9) / seconds;
    double flopsExecuted = (cfg.N > 0) ? static_cast<double>(cfg.N - 1) : 0.0;
    double operationalIntensity = flopsExecuted / bytesRead; // ~ 0.25 FLOP/Byte

    // Format human-readable data volume
    double mbSize = bytesRead / (1024.0 * 1024.0);

    std::cout << std::left << std::setw(20) << cfg.name
              << std::right << std::setw(12) << cfg.N
              << std::setw(11) << std::fixed << std::setprecision(2) << mbSize
              << std::setw(13) << std::fixed << std::setprecision(4) << avgMs
              << std::setw(14) << std::fixed << std::setprecision(2) << effectiveGBs
              << std::setw(14) << std::fixed << std::setprecision(3) << gElementsPerSec
              << std::setw(12) << std::fixed << std::setprecision(3) << operationalIntensity
              << std::endl;

    // Export to CSV
    if (csvWriter) {
        gpu_roofline::BenchmarkRecord rec;
        rec.kernelName = "Reduce_" + cfg.name;
        rec.memoryType = "Streaming";
        rec.problemSize = cfg.N;
        rec.bytesTransferred = bytesRead;
        rec.flosExecuted = flopsExecuted;
        rec.elapsedMillis = avgMs;
        rec.achievedBandwidthGBs = effectiveGBs;
        rec.achievedGflops = (flopsExecuted / 1e9) / seconds;
        rec.operationalIntensity = operationalIntensity;
        csvWriter->appendRecord(rec);
    }

    if (d_temp) CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    try {
        std::cout << "\n==================================================================================================\n"
                  << "                        CUDA PARALLEL FLOAT32 SUM REDUCTION BENCHMARK                             \n"
                  << "==================================================================================================\n";

        gpu_roofline::printDeviceProperties(0);

        std::string csvPath = "results/benchmark_results.csv";
        gpu_roofline::CsvWriter csvWriter(csvPath);

        std::cout << "\n"
                  << std::left << std::setw(20) << "Workload"
                  << std::right << std::setw(12) << "Elements (N)"
                  << std::setw(11) << "Size (MB)"
                  << std::setw(13) << "Latency (ms)"
                  << std::setw(14) << "Eff. BW (GB/s)"
                  << std::setw(14) << "GElem/s"
                  << std::setw(12) << "OI (FLOP/B)"
                  << "\n"
                  << std::string(98, '-') << "\n";

        std::vector<ReduceBenchmarkConfig> configs = {
            // Small arrays (L1/L2 cache residency)
            { "64K-Floats", 65536, 100 },
            { "256K-Floats", 262144, 100 },
            { "1M-Floats", 1048576, 50 },

            // Intermediate / irregular sizes
            { "Irregular-10M", 10000007, 30 },

            // Large arrays (DRAM bandwidth streaming bound)
            { "4M-Floats", 4194304, 30 },
            { "16M-Floats", 16777216, 20 },
            { "64M-Floats", 67108864, 10 }
        };

        for (const auto& cfg : configs) {
            runReduceBenchmark(cfg, &csvWriter);
        }

        std::cout << std::string(98, '-') << "\n"
                  << "[INFO] Benchmark data appended to " << csvPath << std::endl;

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "\n[ERROR] " << ex.what() << std::endl;
        return 1;
    }
}
