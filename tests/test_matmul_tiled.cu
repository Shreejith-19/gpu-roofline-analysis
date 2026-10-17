#include "matmul.cuh"
#include "cuda_utils.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

struct TiledTestCase {
    std::string description;
    int M;
    int N;
    int K;
    int tileDim;
    bool verifyCpu;
};

void runTiledTest(const TiledTestCase& tc) {
    std::cout << "\n======================================================\n"
              << "[TEST] " << tc.description << "\n"
              << "       Matrix Dimensions : M=" << tc.M << ", N=" << tc.N << ", K=" << tc.K << "\n"
              << "       Tile Dimension    : " << tc.tileDim << "x" << tc.tileDim << "\n"
              << "------------------------------------------------------" << std::endl;

    size_t sizeA = static_cast<size_t>(tc.M) * tc.K;
    size_t sizeB = static_cast<size_t>(tc.K) * tc.N;
    size_t sizeC = static_cast<size_t>(tc.M) * tc.N;

    size_t bytesA = sizeA * sizeof(float);
    size_t bytesB = sizeB * sizeof(float);
    size_t bytesC = sizeC * sizeof(float);

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C_naive(sizeC, 0.0f);
    std::vector<float> h_C_tiled(sizeC, 0.0f);
    std::vector<float> h_C_cpu(sizeC, 0.0f);

    gpu_roofline::initMatrixRandom(h_A.data(), sizeA, -1.0f, 1.0f, 101);
    gpu_roofline::initMatrixRandom(h_B.data(), sizeB, -1.0f, 1.0f, 202);

    float *d_A = nullptr, *d_B = nullptr, *d_C_naive = nullptr, *d_C_tiled = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C_naive, bytesC));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

    // 1. Launch Baseline Naive GPU Kernel
    gpu_roofline::launchMatmulNaive(d_A, d_B, d_C_naive, tc.M, tc.N, tc.K, dim3(16, 16));
    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C_naive, bytesC, cudaMemcpyDeviceToHost));

    // 2. Launch Shared-Memory Tiled GPU Kernel
    gpu_roofline::launchMatmulTiled(d_A, d_B, d_C_tiled, tc.M, tc.N, tc.K, tc.tileDim);
    CUDA_CHECK(cudaMemcpy(h_C_tiled.data(), d_C_tiled, bytesC, cudaMemcpyDeviceToHost));

    float atol = 1e-3f + 1e-5f * static_cast<float>(tc.K);
    float rtol = 1e-3f;

    // 3. Verify Tiled GPU vs Naive GPU
    std::cout << "  Comparing Tiled GPU vs Naive GPU..." << std::flush;
    bool passedNaive = gpu_roofline::verifyMatrix(h_C_naive.data(), h_C_tiled.data(), tc.M, tc.N, atol, rtol);
    if (!passedNaive) {
        std::cerr << " FAILED!" << std::endl;
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C_naive));
        CUDA_CHECK(cudaFree(d_C_tiled));
        throw std::runtime_error("Tiled GPU does not match Naive GPU for: " + tc.description);
    }
    std::cout << " MATCHED!" << std::endl;

    // 4. Verify Tiled GPU vs CPU Reference
    if (tc.verifyCpu) {
        std::cout << "  Comparing Tiled GPU vs CPU Reference..." << std::flush;
        gpu_roofline::matmulCpuReference(h_A.data(), h_B.data(), h_C_cpu.data(), tc.M, tc.N, tc.K);
        bool passedCpu = gpu_roofline::verifyMatrix(h_C_cpu.data(), h_C_tiled.data(), tc.M, tc.N, atol, rtol);
        if (!passedCpu) {
            std::cerr << " FAILED!" << std::endl;
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C_naive));
            CUDA_CHECK(cudaFree(d_C_tiled));
            throw std::runtime_error("Tiled GPU does not match CPU reference for: " + tc.description);
        }
        std::cout << " PASSED!" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_tiled));
}

int main() {
    try {
        std::cout << "======================================================\n"
                  << "    SHARED-MEMORY TILED MATRIX MULTIPLICATION TESTS   \n"
                  << "======================================================" << std::endl;

        std::vector<TiledTestCase> testCases = {
            // 1. Square Matrices with 16x16 tile
            { "Square Matrix (64x64x64, tile 16x16)", 64, 64, 64, 16, true },
            { "Square Matrix (256x256x256, tile 16x16)", 256, 256, 256, 16, true },

            // 2. Rectangular Matrices (M != N != K)
            { "Tall-Skinny Matrix (512x64x128, tile 16x16)", 512, 64, 128, 16, true },
            { "Short-Wide Matrix (64x512x256, tile 16x16)", 64, 512, 256, 16, true },

            // 3. Irregular & Non-Multiple Dimensions (Bounds checking for tiles)
            { "Irregular Dimensions (67x113x97, tile 16x16)", 67, 113, 97, 16, true },
            { "Irregular Dimensions (33x17x49, tile 16x16)", 33, 17, 49, 16, true },
            { "Irregular Dimensions (131x73x59, tile 16x16)", 131, 73, 59, 16, true },

            // 4. Configurable Tile Dimension (32x32 tile)
            { "Square Matrix (128x128x128, tile 32x32)", 128, 128, 128, 32, true },
            { "Irregular Dimensions (131x73x59, tile 32x32)", 131, 73, 59, 32, true },

            // 5. Larger Input
            { "Large Matrix (512x512x512, tile 16x16)", 512, 512, 512, 16, true }
        };

        for (const auto& tc : testCases) {
            runTiledTest(tc);
        }

        std::cout << "\n======================================================\n"
                  << "[SUCCESS] All Tiled Matmul verification tests passed!\n"
                  << "======================================================" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "\n[ERROR] " << ex.what() << std::endl;
        return 1;
    }
}
