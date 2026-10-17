#include "matmul.cuh"
#include "cuda_utils.cuh"
#include "common.hpp"

#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

struct BenchmarkConfig {
    std::string name;
    int M;
    int N;
    int K;
    int tileDim;
    int iterations;
};

void runComparisonBenchmark(const BenchmarkConfig& cfg, gpu_roofline::CsvWriter* csvWriter) {
    size_t sizeA = static_cast<size_t>(cfg.M) * cfg.K;
    size_t sizeB = static_cast<size_t>(cfg.K) * cfg.N;
    size_t sizeC = static_cast<size_t>(cfg.M) * cfg.N;

    size_t bytesA = sizeA * sizeof(float);
    size_t bytesB = sizeB * sizeof(float);
    size_t bytesC = sizeC * sizeof(float);

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);

    gpu_roofline::initMatrixRandom(h_A.data(), sizeA, -1.0f, 1.0f, 42);
    gpu_roofline::initMatrixRandom(h_B.data(), sizeB, -1.0f, 1.0f, 84);

    float *d_A = nullptr, *d_B = nullptr, *d_C_naive = nullptr, *d_C_tiled = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C_naive, bytesC));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

    // Warm-up both kernels
    gpu_roofline::launchMatmulNaive(d_A, d_B, d_C_naive, cfg.M, cfg.N, cfg.K, dim3(16, 16));
    gpu_roofline::launchMatmulTiled(d_A, d_B, d_C_tiled, cfg.M, cfg.N, cfg.K, cfg.tileDim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 1. Benchmark Naive Kernel
    gpu_roofline::GpuTimer timerNaive;
    timerNaive.start();
    for (int i = 0; i < cfg.iterations; ++i) {
        gpu_roofline::launchMatmulNaive(d_A, d_B, d_C_naive, cfg.M, cfg.N, cfg.K, dim3(16, 16));
    }
    timerNaive.stop();
    float naiveTotalMs = timerNaive.elapsedMillis();
    float naiveAvgMs = naiveTotalMs / cfg.iterations;

    // 2. Benchmark Tiled Kernel
    gpu_roofline::GpuTimer timerTiled;
    timerTiled.start();
    for (int i = 0; i < cfg.iterations; ++i) {
        gpu_roofline::launchMatmulTiled(d_A, d_B, d_C_tiled, cfg.M, cfg.N, cfg.K, cfg.tileDim);
    }
    timerTiled.stop();
    float tiledTotalMs = timerTiled.elapsedMillis();
    float tiledAvgMs = tiledTotalMs / cfg.iterations;

    // FLOPs & Bandwidth Calculations
    double totalFlops = 2.0 * static_cast<double>(cfg.M) * cfg.N * cfg.K;
    double naiveGflops = (totalFlops / 1e9) / (naiveAvgMs / 1000.0);
    double tiledGflops = (totalFlops / 1e9) / (tiledAvgMs / 1000.0);
    double speedup = naiveAvgMs / tiledAvgMs;

    // Cold DRAM theoretical footprint
    double coldBytes = static_cast<double>(bytesA + bytesB + bytesC);
    double algorithmicOI = totalFlops / coldBytes;

    // Naive DRAM traffic: each thread loads K from A and K from B
    double naiveBytes = 2.0 * static_cast<double>(cfg.M) * cfg.N * cfg.K * sizeof(float);
    double naiveOI = totalFlops / naiveBytes; // 0.25 FLOP/Byte

    // Tiled DRAM traffic: each block loads tiles of A and B
    double tiledBytes = (static_cast<double>(cfg.M) * cfg.N * cfg.K * 2.0 * sizeof(float)) / cfg.tileDim;
    double tiledOI = totalFlops / tiledBytes; // ~ TILE_DIM / 4 FLOP/Byte

    // Print Formatted Output Row
    std::cout << std::left << std::setw(24) << cfg.name
              << std::right << std::setw(15) << (std::to_string(cfg.M) + "x" + std::to_string(cfg.N) + "x" + std::to_string(cfg.K))
              << std::setw(12) << std::fixed << std::setprecision(3) << naiveAvgMs
              << std::setw(12) << std::fixed << std::setprecision(3) << tiledAvgMs
              << std::setw(13) << std::fixed << std::setprecision(2) << naiveGflops
              << std::setw(13) << std::fixed << std::setprecision(2) << tiledGflops
              << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x"
              << std::setw(12) << std::fixed << std::setprecision(2) << algorithmicOI
              << std::endl;

    // Export to CSV if writer provided
    if (csvWriter) {
        gpu_roofline::BenchmarkRecord recNaive;
        recNaive.kernelName = "Naive_" + std::to_string(cfg.M) + "x" + std::to_string(cfg.N);
        recNaive.memoryType = "Global";
        recNaive.problemSize = static_cast<size_t>(cfg.M) * cfg.N;
        recNaive.bytesTransferred = coldBytes;
        recNaive.flosExecuted = totalFlops;
        recNaive.elapsedMillis = naiveAvgMs;
        recNaive.achievedBandwidthGBs = (coldBytes / 1e9) / (naiveAvgMs / 1000.0);
        recNaive.achievedGflops = naiveGflops;
        recNaive.operationalIntensity = algorithmicOI;
        csvWriter->appendRecord(recNaive);

        gpu_roofline::BenchmarkRecord recTiled;
        recTiled.kernelName = "Tiled" + std::to_string(cfg.tileDim) + "_" + std::to_string(cfg.M) + "x" + std::to_string(cfg.N);
        recTiled.memoryType = "Shared";
        recTiled.problemSize = static_cast<size_t>(cfg.M) * cfg.N;
        recTiled.bytesTransferred = coldBytes;
        recTiled.flosExecuted = totalFlops;
        recTiled.elapsedMillis = tiledAvgMs;
        recTiled.achievedBandwidthGBs = (coldBytes / 1e9) / (tiledAvgMs / 1000.0);
        recTiled.achievedGflops = tiledGflops;
        recTiled.operationalIntensity = algorithmicOI;
        csvWriter->appendRecord(recTiled);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_tiled));
}

int main() {
    try {
        std::cout << "\n=======================================================================================================\n"
                  << "                        CUDA MATRIX MULTIPLICATION: NAIVE vs TILED BENCHMARK                           \n"
                  << "=======================================================================================================\n";

        gpu_roofline::printDeviceProperties(0);

        std::string csvPath = "results/benchmark_results.csv";
        gpu_roofline::CsvWriter csvWriter(csvPath);
        csvWriter.writeHeader();

        std::cout << "\n"
                  << std::left << std::setw(24) << "Workload"
                  << std::right << std::setw(15) << "Dimensions"
                  << std::setw(12) << "Naive (ms)"
                  << std::setw(12) << "Tiled (ms)"
                  << std::setw(13) << "Naive GFLOP"
                  << std::setw(13) << "Tiled GFLOP"
                  << std::setw(11) << "Speedup"
                  << std::setw(12) << "Alg. OI"
                  << "\n"
                  << std::string(103, '-') << "\n";

        std::vector<BenchmarkConfig> configs = {
            // Small matrices (Latency & launch overhead dominated)
            { "Tiny Square", 64, 64, 64, 16, 20 },
            { "Small Square", 128, 128, 128, 16, 20 },

            // Rectangular matrices
            { "Tall-Skinny", 512, 64, 128, 16, 20 },
            { "Short-Wide", 64, 512, 256, 16, 20 },

            // Irregular non-multiple dimensions
            { "Irregular-1", 67, 113, 97, 16, 20 },
            { "Irregular-2", 131, 73, 59, 16, 20 },
            { "Irregular-3", 257, 255, 253, 16, 20 },

            // Medium & Large matrices (Memory bandwidth vs compute capacity)
            { "Medium Square", 256, 256, 256, 16, 20 },
            { "Large Square-512", 512, 512, 512, 16, 10 },
            { "Large Square-1024", 1024, 1024, 1024, 16, 5 },
            { "Large Square-2048", 2048, 2048, 2048, 16, 3 }
        };

        for (const auto& cfg : configs) {
            runComparisonBenchmark(cfg, &csvWriter);
        }

        std::cout << std::string(103, '-') << "\n"
                  << "[INFO] Results exported to " << csvPath << std::endl;

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "\n[ERROR] " << ex.what() << std::endl;
        return 1;
    }
}
