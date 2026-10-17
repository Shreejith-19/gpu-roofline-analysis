#include "reduce.cuh"
#include "cuda_utils.cuh"

#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

struct ReduceTestCase {
    std::string name;
    size_t N;
};

void runReduceTestCase(const ReduceTestCase& tc) {
    std::cout << "[TEST] " << std::left << std::setw(28) << tc.name
              << " (N=" << std::right << std::setw(10) << tc.N << ") ... " << std::flush;

    if (tc.N == 0) {
        // 1. Test host API for N = 0
        float resHost = gpu_roofline::reduceSum(nullptr, 0);
        if (resHost != 0.0f) {
            std::cerr << "FAILED! Expected 0.0f for N=0, got " << resHost << std::endl;
            throw std::runtime_error("reduceSum failed on N=0.");
        }

        // 2. Test device-level launchReduceSum for N = 0
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        float initVal = 999.0f;
        CUDA_CHECK(cudaMemcpy(d_out, &initVal, sizeof(float), cudaMemcpyHostToDevice));
        gpu_roofline::launchReduceSum(nullptr, d_out, 0);
        float resDev = -1.0f;
        CUDA_CHECK(cudaMemcpy(&resDev, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_out));

        if (resDev != 0.0f) {
            std::cerr << "FAILED! Expected 0.0f for launchReduceSum N=0, got " << resDev << std::endl;
            throw std::runtime_error("launchReduceSum failed on N=0.");
        }

        std::cout << "PASSED (Zero handled cleanly)" << std::endl;
        return;
    }

    // Host buffer allocation
    std::vector<float> h_input(tc.N);
    // Initialize with bounded deterministic floats in range [-0.5, 0.5]
    // to preserve precision during large sum accumulations
    for (size_t i = 0; i < tc.N; ++i) {
        h_input[i] = static_cast<float>(static_cast<int>(i % 1000) - 500) / 500.0f;
    }

    // CPU Reference sum (double precision accumulator)
    float cpuSum = gpu_roofline::reduceSumCpuReference(h_input.data(), tc.N, true);

    // Device allocation
    float* d_input = nullptr;
    size_t bytes = tc.N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // Execute GPU reduction
    float gpuSum = gpu_roofline::reduceSum(d_input, tc.N);

    // Verify tolerance
    float rtol = 1e-4f;
    bool passed = gpu_roofline::verifyReduction(cpuSum, gpuSum, tc.N, rtol);

    CUDA_CHECK(cudaFree(d_input));

    if (passed) {
        std::cout << "PASSED (CPU=" << std::fixed << std::setprecision(4) << cpuSum 
                  << ", GPU=" << gpuSum << ")" << std::endl;
    } else {
        std::cerr << "FAILED!" << std::endl;
        throw std::runtime_error("Reduction mismatch for test: " + tc.name);
    }
}

int main() {
    try {
        std::cout << "======================================================\n"
                  << "        PARALLEL FLOAT32 SUM REDUCTION TESTS          \n"
                  << "======================================================" << std::endl;

        std::vector<ReduceTestCase> testCases = {
            { "Zero Elements (N=0 Edge Case)", 0 },
            { "Single Element (N=1)", 1 },
            { "Two Elements (N=2)", 2 },
            { "Sub-Warp Size (N=17)", 17 },
            { "Exact Warp Size (N=32)", 32 },
            { "Small Irregular (N=67)", 67 },
            { "Sub-Block Size (N=128)", 128 },
            { "Exact Block Size (N=256)", 256 },
            { "Slightly Over Block (N=257)", 257 },
            { "Multi-Block Power of 2 (N=1024)", 1024 },
            { "Prime Element Count (N=10007)", 10007 },
            { "Arbitrary Non-Power-of-2 (N=65537)", 65537 },
            { "Medium Irregular (N=123456)", 123456 },
            { "One Million Elements (N=1M)", 1000000 },
            { "Large Power-of-2 (N=16M Elements)", 16777216 }
        };

        for (const auto& tc : testCases) {
            runReduceTestCase(tc);
        }

        std::cout << "\n======================================================\n"
                  << "[SUCCESS] All Parallel Sum Reduction tests passed!\n"
                  << "======================================================" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "\n[ERROR] " << ex.what() << std::endl;
        return 1;
    }
}
