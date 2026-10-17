#include "reduce.cuh"
#include "cuda_utils.cuh"

#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <cmath>
#include <limits>

void testEmptyArray() {
    std::cout << "[TEST] Empty Array (N=0) ... " << std::flush;
    
    // Test synchronous host API
    float resHost = gpu_roofline::reduceMax(nullptr, 0);
    if (!std::isinf(resHost) || resHost > 0) {
        std::cerr << "FAILED! Expected -INFINITY for N=0, got " << resHost << std::endl;
        throw std::runtime_error("reduceMax failed for N=0");
    }

    // Test asynchronous device launch API
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    float dummy = 1234.0f;
    CUDA_CHECK(cudaMemcpy(d_out, &dummy, sizeof(float), cudaMemcpyHostToDevice));
    
    gpu_roofline::launchReduceMax(nullptr, d_out, 0);
    float resDev = 0.0f;
    CUDA_CHECK(cudaMemcpy(&resDev, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    if (!std::isinf(resDev) || resDev > 0) {
        std::cerr << "FAILED! Expected -INFINITY for launchReduceMax N=0, got " << resDev << std::endl;
        throw std::runtime_error("launchReduceMax failed for N=0");
    }

    std::cout << "PASSED (-INFINITY returned)" << std::endl;
}

void testSingleElement() {
    std::cout << "[TEST] Single-Element Arrays (N=1) ... " << std::flush;

    float vals[] = { 42.5f, -999.0f, 0.0f, 1e6f };
    for (float v : vals) {
        float* d_in = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, &v, sizeof(float), cudaMemcpyHostToDevice));

        float res = gpu_roofline::reduceMax(d_in, 1);
        CUDA_CHECK(cudaFree(d_in));

        if (res != v) {
            std::cerr << "FAILED! Expected " << v << ", got " << res << std::endl;
            throw std::runtime_error("Single element test failed");
        }
    }
    std::cout << "PASSED" << std::endl;
}

void testNegativeValues() {
    std::cout << "[TEST] All Negative Values ... " << std::flush;

    constexpr size_t N = 10000;
    std::vector<float> h_input(N);
    // Values strictly between -10000.0 and -50.0
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = -10000.0f + static_cast<float>(i % 9950);
    }
    // Plant maximum at an arbitrary index
    h_input[4321] = -50.0f;

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float res = gpu_roofline::reduceMax(d_in, N);
    CUDA_CHECK(cudaFree(d_in));

    float cpuRef = gpu_roofline::reduceMaxCpuReference(h_input.data(), N);
    if (!gpu_roofline::verifyMaxReduction(cpuRef, res) || res != -50.0f) {
        std::cerr << "FAILED! Expected -50.0f, got " << res << " (CPU ref=" << cpuRef << ")" << std::endl;
        throw std::runtime_error("Negative values test failed");
    }
    std::cout << "PASSED (Max=" << res << ")" << std::endl;
}

void testRepeatedMaxima() {
    std::cout << "[TEST] Repeated Maxima ... " << std::flush;

    constexpr size_t N = 65536;
    constexpr float PEAK_VAL = 777.7f;
    std::vector<float> h_input(N, 10.0f);

    // Scatter repeated peak values across different warps and blocks
    h_input[0] = PEAK_VAL;
    h_input[31] = PEAK_VAL;
    h_input[255] = PEAK_VAL;
    h_input[10000] = PEAK_VAL;
    h_input[N - 1] = PEAK_VAL;

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float res = gpu_roofline::reduceMax(d_in, N);
    CUDA_CHECK(cudaFree(d_in));

    if (res != PEAK_VAL) {
        std::cerr << "FAILED! Expected " << PEAK_VAL << ", got " << res << std::endl;
        throw std::runtime_error("Repeated maxima test failed");
    }
    std::cout << "PASSED (Max=" << res << ")" << std::endl;
}

void testNanPolicy() {
    std::cout << "[TEST] NaN Missing-Value Policy ... " << std::flush;

    constexpr size_t N = 1000;
    std::vector<float> h_input(N, 5.0f);
    // Insert NaNs at various boundaries
    float nanVal = std::numeric_limits<float>::quiet_NaN();
    h_input[0] = nanVal;
    h_input[15] = nanVal;
    h_input[256] = nanVal;
    h_input[N - 1] = nanVal;
    h_input[500] = 99.0f; // True max

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float res = gpu_roofline::reduceMax(d_in, N);
    CUDA_CHECK(cudaFree(d_in));

    float cpuRef = gpu_roofline::reduceMaxCpuReference(h_input.data(), N);
    if (!gpu_roofline::verifyMaxReduction(cpuRef, res) || res != 99.0f) {
        std::cerr << "FAILED! Expected 99.0f, got " << res << " (CPU ref=" << cpuRef << ")" << std::endl;
        throw std::runtime_error("NaN policy test failed");
    }
    std::cout << "PASSED (Ignored NaNs, Max=" << res << ")" << std::endl;
}

void testIrregularAndLarge(size_t N, const std::string& label) {
    std::cout << "[TEST] " << std::left << std::setw(25) << label 
              << " (N=" << std::right << std::setw(10) << N << ") ... " << std::flush;

    std::vector<float> h_input(N);
    for (size_t i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(static_cast<int>(i % 50000) - 25000);
    }
    // Plant unique maximum
    size_t targetIdx = N / 3;
    h_input[targetIdx] = 88888.0f;

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float res = gpu_roofline::reduceMax(d_in, N);
    CUDA_CHECK(cudaFree(d_in));

    float cpuRef = gpu_roofline::reduceMaxCpuReference(h_input.data(), N);
    if (!gpu_roofline::verifyMaxReduction(cpuRef, res) || res != 88888.0f) {
        std::cerr << "FAILED! Expected 88888.0f, got " << res << " (CPU ref=" << cpuRef << ")" << std::endl;
        throw std::runtime_error("Irregular/Large test failed for: " + label);
    }
    std::cout << "PASSED (Max=" << res << ")" << std::endl;
}

int main() {
    try {
        std::cout << "======================================================\n"
                  << "        PARALLEL FLOAT32 MAX REDUCTION TESTS          \n"
                  << "======================================================" << std::endl;

        testEmptyArray();
        testSingleElement();
        testNegativeValues();
        testRepeatedMaxima();
        testNanPolicy();

        // Irregular and non-power-of-two lengths
        testIrregularAndLarge(17, "Sub-Warp Irregular");
        testIrregularAndLarge(67, "Small Irregular");
        testIrregularAndLarge(255, "Sub-Block Boundary");
        testIrregularAndLarge(257, "Block Overflow Boundary");
        testIrregularAndLarge(1007, "Prime Length");
        testIrregularAndLarge(65537, "65K+1 Non-Multiple");
        testIrregularAndLarge(123456, "Arbitrary Irregular");

        // Large arrays
        testIrregularAndLarge(1000000, "One Million Floats");
        testIrregularAndLarge(16777216, "16M Large Buffer");

        std::cout << "\n======================================================\n"
                  << "[SUCCESS] All Parallel Max Reduction tests passed!\n"
                  << "======================================================" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "\n[ERROR] " << ex.what() << std::endl;
        return 1;
    }
}
