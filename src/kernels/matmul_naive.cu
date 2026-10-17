#include "matmul.cuh"
#include "cuda_utils.cuh"

#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <iomanip>

namespace gpu_roofline {

/**
 * @brief CUDA kernel for baseline naive matrix multiplication C = A * B.
 * 
 * Thread Mapping:
 *   - Each thread computes exactly one element C[row, col].
 *   - threadIdx.x maps to column index (col = blockIdx.x * blockDim.x + threadIdx.x).
 *   - threadIdx.y maps to row index (row = blockIdx.y * blockDim.y + threadIdx.y).
 * 
 * Memory Access Pattern:
 *   - A[row * K + k]: Threads with consecutive threadIdx.x share the exact same row index.
 *     Thus, all threads in a warp read the identical element of A, which can be broadcast via L1/L2 cache.
 *   - B[k * N + col]: Threads with consecutive threadIdx.x access consecutive columns in row-major order.
 *     This yields coalesced 128-byte DRAM transactions across threads in a warp.
 *   - C[row * N + col]: Coalesced write upon loop completion.
 * 
 * Boundary Conditions:
 *   - If M or N is not an exact multiple of the block dimension, boundary threads are safely masked.
 */
__global__ void matmulNaiveKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launchMatmulNaive(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    dim3 blockDim,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("Matrix dimensions (M, N, K) must be positive integers.");
    }
    if (blockDim.x == 0 || blockDim.y == 0 || (blockDim.x * blockDim.y * blockDim.z) > 1024) {
        throw std::invalid_argument("Invalid block dimension: total threads per block must be in [1, 1024].");
    }

    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y,
        1
    );

    matmulNaiveKernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_LAST_ERROR();
}

void matmulCpuReference(
    const float* h_A,
    const float* h_B,
    float* h_C,
    int M,
    int N,
    int K
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C[i * N + j] = sum;
        }
    }
}

bool verifyMatrix(
    const float* ref,
    const float* test,
    int M,
    int N,
    float atol,
    float rtol
) {
    size_t total = static_cast<size_t>(M) * N;
    size_t maxErrorsReported = 10;
    size_t errorCount = 0;
    float maxDiff = 0.0f;
    float maxRelError = 0.0f;

    for (size_t idx = 0; idx < total; ++idx) {
        float r = ref[idx];
        float t = test[idx];
        float diff = std::fabs(r - t);
        float tol = atol + rtol * std::fabs(r);

        if (diff > maxDiff) {
            maxDiff = diff;
        }

        if (std::fabs(r) > 1e-6f) {
            float rel = diff / std::fabs(r);
            if (rel > maxRelError) maxRelError = rel;
        }

        if (diff > tol) {
            if (errorCount < maxErrorsReported) {
                int row = static_cast<int>(idx / N);
                int col = static_cast<int>(idx % N);
                std::cerr << "  [MISMATCH] at (" << row << ", " << col << "): "
                          << "CPU=" << r << ", GPU=" << t 
                          << ", diff=" << diff << ", allowed_tol=" << tol << "\n";
            }
            errorCount++;
        }
    }

    if (errorCount > 0) {
        std::cerr << "  [VERIFY FAILED] " << errorCount << " / " << total 
                  << " elements failed tolerance check (" 
                  << std::fixed << std::setprecision(2) << (100.0 * errorCount / total) << "%). "
                  << "Max abs diff: " << maxDiff 
                  << ", Max rel error: " << maxRelError << "\n";
        return false;
    }

    return true;
}

void initMatrixRandom(float* mat, size_t totalElements, float minVal, float maxVal, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(minVal, maxVal);

    for (size_t i = 0; i < totalElements; ++i) {
        mat[i] = dist(gen);
    }
}

} // namespace gpu_roofline
