#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace gpu_roofline {

/**
 * @brief Launches the baseline naive row-major matrix multiplication kernel C = A * B.
 * 
 * Each thread computes exactly one element of the output matrix C[row, col]
 * by executing an inner accumulation loop over K with direct global memory loads.
 * 
 * Matrix layout is row-major:
 *   A is M x K (A[i, k] = A[i * K + k])
 *   B is K x N (B[k, j] = B[k * N + j])
 *   C is M x N (C[i, j] = C[i * N + j])
 * 
 * @param d_A Pointer to device memory for input matrix A [M x K].
 * @param d_B Pointer to device memory for input matrix B [K x N].
 * @param d_C Pointer to device memory for output matrix C [M x N].
 * @param M Number of rows in A and C.
 * @param N Number of columns in B and C.
 * @param K Number of columns in A and rows in B.
 * @param blockDim 2D thread block dimensions (default: 16 x 16).
 * @param stream CUDA stream for asynchronous launch (default: 0).
 */
void launchMatmulNaive(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    dim3 blockDim = dim3(16, 16),
    cudaStream_t stream = 0
);

/**
 * @brief CPU reference implementation of single-precision matrix multiplication C = A * B.
 * Used for correctness validation against GPU results.
 * 
 * @param h_A Host pointer to input matrix A [M x K].
 * @param h_B Host pointer to input matrix B [K x N].
 * @param h_C Host pointer to output matrix C [M x N].
 * @param M Number of rows in A and C.
 * @param N Number of columns in B and C.
 * @param K Number of columns in A and rows in B.
 */
void matmulCpuReference(
    const float* h_A,
    const float* h_B,
    float* h_C,
    int M,
    int N,
    int K
);

/**
 * @brief Validates GPU matrix results against CPU reference within given tolerance.
 * Tolerance check: |ref - test| <= atol + rtol * |ref|
 * 
 * @param ref Host pointer to reference matrix [M x N].
 * @param test Host pointer to test matrix [M x N].
 * @param M Number of rows.
 * @param N Number of columns.
 * @param atol Absolute tolerance (default: 1e-4f).
 * @param rtol Relative tolerance (default: 1e-3f).
 * @return true if all elements are within tolerance, false otherwise.
 */
bool verifyMatrix(
    const float* ref,
    const float* test,
    int M,
    int N,
    float atol = 1e-4f,
    float rtol = 1e-3f
);

/**
 * @brief Fills a float matrix with deterministic pseudorandom or uniform test data.
 */
void initMatrixRandom(float* mat, size_t totalElements, float minVal = -1.0f, float maxVal = 1.0f, unsigned int seed = 42);

} // namespace gpu_roofline
