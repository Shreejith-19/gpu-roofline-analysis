#include "matmul.cuh"
#include "cuda_utils.cuh"

#include <iostream>
#include <stdexcept>

namespace gpu_roofline {

/**
 * @brief CUDA kernel for shared-memory tiled matrix multiplication C = A * B.
 * 
 * Each thread block of dimension (TILE_DIM, TILE_DIM) computes a sub-matrix of C.
 * Threads cooperatively stage sub-matrices (tiles) of A and B from global memory
 * into on-chip __shared__ memory.
 * 
 * Data Reuse & Bandwidth Amplification:
 *   - Each element loaded into s_A[ty][tx] is reused TILE_DIM times by all threads in the same row (ty).
 *   - Each element loaded into s_B[ty][tx] is reused TILE_DIM times by all threads in the same col (tx).
 *   - Global memory traffic is reduced by a factor of roughly TILE_DIM compared to the naive kernel.
 * 
 * Barrier Synchronization:
 *   - Barrier 1: __syncthreads() after loading s_A and s_B ensures all threads in the block have
 *                populated their shared memory locations before any thread consumes them.
 *   - Barrier 2: __syncthreads() after multiplying ensures all threads have completed computation
 *                with the current tile before any thread overwrites s_A and s_B with the next tile.
 * 
 * Bounds Safety:
 *   - Zero-padding is applied when tile coordinates exceed M, N, or K.
 */
template <int TILE_DIM>
__global__ void matmulTiledKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    // 2D shared memory tiles for sub-matrices of A and B
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = blockIdx.x * TILE_DIM + tx;
    const int row = blockIdx.y * TILE_DIM + ty;

    float sum = 0.0f;

    // Number of phase tiles required to traverse K
    const int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        // 1. Cooperative load of matrix A into shared memory s_A
        // Element index in A: row = 'row', col = 't * TILE_DIM + tx'
        const int aCol = t * TILE_DIM + tx;
        if (row < M && aCol < K) {
            s_A[ty][tx] = A[row * K + aCol];
        } else {
            s_A[ty][tx] = 0.0f; // Safe zero padding for boundary threads
        }

        // 2. Cooperative load of matrix B into shared memory s_B
        // Element index in B: row = 't * TILE_DIM + ty', col = 'col'
        const int bRow = t * TILE_DIM + ty;
        if (bRow < K && col < N) {
            s_B[ty][tx] = B[bRow * N + col];
        } else {
            s_B[ty][tx] = 0.0f; // Safe zero padding for boundary threads
        }

        // Barrier 1: Wait until all threads in this block finish loading their tile elements
        __syncthreads();

        // 3. Multiply-accumulate over the staged tiles in shared memory
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // Barrier 2: Wait until all threads in this block finish reading before next tile overwrite
        __syncthreads();
    }

    // 4. Store accumulated sum to global memory C with boundary check
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launchMatmulTiled(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int tileDim,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("Matrix dimensions (M, N, K) must be positive integers.");
    }

    // Query active device and validate shared memory allocation
    int deviceId = 0;
    CUDA_CHECK(cudaGetDevice(&deviceId));

    int maxSharedMem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlock, deviceId));

    size_t requiredSharedMem = 2 * static_cast<size_t>(tileDim) * tileDim * sizeof(float);
    if (requiredSharedMem > static_cast<size_t>(maxSharedMem)) {
        throw std::runtime_error("Requested tile dimension requires " + std::to_string(requiredSharedMem) +
                                 " bytes of shared memory, exceeding device limit of " +
                                 std::to_string(maxSharedMem) + " bytes.");
    }

    dim3 block(tileDim, tileDim);
    dim3 grid(
        (N + tileDim - 1) / tileDim,
        (M + tileDim - 1) / tileDim,
        1
    );

    if (tileDim == 16) {
        matmulTiledKernel<16><<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    } else if (tileDim == 32) {
        matmulTiledKernel<32><<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    } else {
        throw std::invalid_argument("Unsupported tileDim: " + std::to_string(tileDim) +
                                    ". Supported tile dimensions are 16 and 32.");
    }

    CUDA_CHECK_LAST_ERROR();
}

} // namespace gpu_roofline
