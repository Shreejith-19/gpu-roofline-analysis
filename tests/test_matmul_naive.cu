#include "matmul.cuh"
#include "cuda_utils.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

struct TestCase {
    std::string description;
    int M;
    int N;
    int K;
    dim3 blockDim;
    bool runCpuVerify; // For very large matrices, we can optionally skip CPU to save host runtime
};

void runTestCase(const TestCase& tc) {
    std::cout << "\n======================================================\n"
              << "[TEST] " << tc.description << "\n"
              << "       Matrix Dimensions : M=" << tc.M << ", N=" << tc.N << ", K=" << tc.K << "\n"
              << "       Block Dimensions  : (" << tc.blockDim.x << ", " << tc.blockDim.y << ")\n"
              << "------------------------------------------------------" << std::endl;

    size_t sizeA = static_cast<size_t>(tc.M) * tc.K;
    size_t sizeB = static_cast<size_t>(tc.K) * tc.N;
    size_t sizeC = static_cast<size_t>(tc.M) * tc.N;

    size_t bytesA = sizeA * sizeof(float);
    size_t bytesB = sizeB * sizeof(float);
    size_t bytesC = sizeC * sizeof(float);

    // Host allocations
    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C_gpu(sizeC, 0.0f);
    std::vector<float> h_C_cpu(sizeC, 0.0f);

    // Initialize with bounded random values [-1.0, 1.0] to control numerical error accumulation
    gpu_roofline::initMatrixRandom(h_A.data(), sizeA, -1.0f, 1.0f, 101);
    gpu_roofline::initMatrixRandom(h_B.data(), sizeB, -1.0f, 1.0f, 202);

    // Device allocations
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

    // Warm-up launch
    gpu_roofline::launchMatmulNaive(d_A, d_B, d_C, tc.M, tc.N, tc.K, tc.blockDim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed launch
    gpu_roofline::GpuTimer timer;
    timer.start();
    
    constexpr int NUM_ITERATIONS = 5;
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        gpu_roofline::launchMatmulNaive(d_A, d_B, d_C, tc.M, tc.N, tc.K, tc.blockDim);
    }
    timer.stop();
    float totalMs = timer.elapsedMillis();
    float avgMs = totalMs / NUM_ITERATIONS;

    // Copy GPU result back
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    // Calculate metrics
    // Total FLOPs = 2 * M * N * K (1 Multiply + 1 Add per element inner loop)
    double totalFlops = 2.0 * static_cast<double>(tc.M) * tc.N * tc.K;
    double gflops = (totalFlops / 1e9) / (avgMs / 1000.0);

    // Theoretical minimal global bytes transferred = (M*K + K*N + M*N) * sizeof(float)
    double totalBytes = static_cast<double>(bytesA + bytesB + bytesC);
    double bandwidthGBs = (totalBytes / 1e9) / (avgMs / 1000.0);
    double operationalIntensity = totalFlops / totalBytes;

    std::cout << "  Execution Time (avg)   : " << std::fixed << std::setprecision(3) << avgMs << " ms\n"
              << "  Throughput             : " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s\n"
              << "  Theoretical DRAM BW    : " << std::fixed << std::setprecision(2) << bandwidthGBs << " GB/s\n"
              << "  Operational Intensity  : " << std::fixed << std::setprecision(3) << operationalIntensity << " FLOP/Byte\n";

    // Correctness verification
    if (tc.runCpuVerify) {
        std::cout << "  Running CPU reference verification..." << std::flush;
        gpu_roofline::matmulCpuReference(h_A.data(), h_B.data(), h_C_cpu.data(), tc.M, tc.N, tc.K);

        // Absolute tolerance scaled by K to account for floating-point accumulation rounding
        float atol = 1e-3f + 1e-5f * static_cast<float>(tc.K);
        float rtol = 1e-3f;

        bool passed = gpu_roofline::verifyMatrix(h_C_cpu.data(), h_C_gpu.data(), tc.M, tc.N, atol, rtol);
        if (passed) {
            std::cout << " PASSED!" << std::endl;
        } else {
            std::cerr << " FAILED!" << std::endl;
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            throw std::runtime_error("Matrix multiplication verification failed for: " + tc.description);
        }
    } else {
        std::cout << "  [INFO] CPU reference skipped for large problem size." << std::endl;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    try {
        std::cout << "======================================================\n"
                  << "  BASELINE NAIVE MATRIX MULTIPLICATION TEST SUITE     \n"
                  << "======================================================" << std::endl;

        std::vector<TestCase> testCases = {
            // 1. Square Matrices
            { "Small Square Matrix (64x64x64)", 64, 64, 64, dim3(16, 16), true },
            { "Medium Square Matrix (256x256x256)", 256, 256, 256, dim3(16, 16), true },

            // 2. Rectangular Matrices (M != N != K)
            { "Tall-Skinny Matrix (512x64x128)", 512, 64, 128, dim3(16, 16), true },
            { "Short-Wide Matrix (64x512x256)", 64, 512, 256, dim3(16, 16), true },

            // 3. Irregular & Non-Multiple Dimensions (Bounds checking)
            { "Irregular Dimensions (67x113x97, block 16x16)", 67, 113, 97, dim3(16, 16), true },
            { "Irregular Dimensions (33x17x49, block 8x8)", 33, 17, 49, dim3(8, 8), true },
            { "Irregular Dimensions (131x73x59, asymmetric block 32x8)", 131, 73, 59, dim3(32, 8), true },

            // 4. Configurable Block Sizes (Same problem, different 2D thread blocks)
            { "Block Size Sweep: 8x8 (128x128x128)", 128, 128, 128, dim3(8, 8), true },
            { "Block Size Sweep: 16x16 (128x128x128)", 128, 128, 128, dim3(16, 16), true },
            { "Block Size Sweep: 32x32 (128x128x128)", 128, 128, 128, dim3(32, 32), true },

            // 5. Larger Inputs (Performance & memory scale)
            { "Large Matrix (512x512x512, block 16x16)", 512, 512, 512, dim3(16, 16), true },
            { "Large Matrix (1024x1024x1024, block 16x16)", 1024, 1024, 1024, dim3(16, 16), false }
        };

        for (const auto& tc : testCases) {
            runTestCase(tc);
        }

        std::cout << "\n======================================================\n"
                  << "[SUCCESS] All Baseline Naive Matmul tests passed!\n"
                  << "======================================================" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "\n[ERROR] " << ex.what() << std::endl;
        return 1;
    }
}
