#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace gpu_roofline {

/**
 * @brief Asynchronously launches parallel sum reduction on device.
 * 
 * Computes: *d_output = sum(d_input[0 ... N-1])
 * 
 * Employs a two-pass reduction:
 *   - Pass 1: G blocks execute grid-stride loops and shared-memory tree reductions
 *             to produce G partial sums into d_temp.
 *   - Pass 2: A single block reduces the G partial sums to the final scalar *d_output.
 * 
 * Handles arbitrary N, including N = 0 explicitly (stores 0.0f to *d_output).
 * 
 * @param d_input Pointer to device array [N].
 * @param d_output Pointer to device scalar to store the final reduced sum.
 * @param N Number of elements.
 * @param d_temp Optional pointer to device temporary storage for partial sums.
 *               If nullptr, scratch memory is dynamically allocated.
 * @param stream CUDA stream (default: 0).
 */
void launchReduceSum(
    const float* d_input,
    float* d_output,
    size_t N,
    float* d_temp = nullptr,
    cudaStream_t stream = 0
);

/**
 * @brief Synchronous host convenience API for float32 sum reduction.
 * 
 * Handles arbitrary N, including N = 0 (returns 0.0f immediately without kernel launch).
 * 
 * @param d_input Device pointer to input array [N].
 * @param N Number of elements.
 * @param d_temp Optional pointer to device temporary storage.
 * @param stream CUDA stream (default: 0).
 * @return float The computed sum.
 */
float reduceSum(
    const float* d_input,
    size_t N,
    float* d_temp = nullptr,
    cudaStream_t stream = 0
);

/**
 * @brief Computes required temporary storage size in bytes for a given N.
 */
size_t getReduceSumTempStorageBytes(size_t N, int blockSize = 256);

/**
 * @brief Returns the number of blocks used in Pass 1.
 */
int getReduceSumNumBlocks(size_t N, int blockSize = 256);

/**
 * @brief CPU reference implementation of sum reduction.
 * 
 * @param h_input Pointer to host array [N].
 * @param N Number of elements.
 * @param useDoublePrecision If true, uses double accumulator to minimize sequential rounding drift.
 * @return float Computed sum.
 */
float reduceSumCpuReference(
    const float* h_input,
    size_t N,
    bool useDoublePrecision = true
);

/**
 * @brief Verifies reduction result against CPU reference within numerical tolerance.
 * 
 * Due to non-associativity of floating-point addition, pairwise/tree GPU addition
 * accumulates rounding error differently than sequential CPU addition.
 * 
 * @param ref CPU reference sum.
 * @param test GPU test sum.
 * @param N Number of elements reduced.
 * @param rtol Relative tolerance (default: 1e-4f).
 * @return true if within tolerance, false otherwise.
 */
bool verifyReduction(
    float ref,
    float test,
    size_t N,
    float rtol = 1e-4f
);

// ============================================================================
// PARALLEL FLOAT32 MAX REDUCTION
// ============================================================================

/**
 * @brief Asynchronously launches parallel max reduction on device.
 * 
 * Computes: *d_output = max(d_input[0 ... N-1])
 * 
 * Behavioral & Numerical Policies:
 *   - Empty Input (N = 0):
 *     Explicitly handled without launching kernels. Writes -INFINITY to *d_output
 *     (-std::numeric_limits<float>::infinity()), which is the mathematical identity
 *     element for the maximum operator (max(x, -inf) = x).
 *   - NaN Policy:
 *     Adheres to the IEEE 754-2008 maxNum / CUDA fmaxf policy (missing-value semantics):
 *     fmaxf(x, NaN) = x and fmaxf(NaN, x) = x.
 *     If the array consists entirely of NaNs, the reduction yields -INFINITY.
 * 
 * Employs a two-pass reduction:
 *   - Pass 1: G blocks execute grid-stride loops and shared-memory tree reductions
 *             to produce G partial maxima into d_temp.
 *   - Pass 2: A single block reduces the G partial maxima to the final scalar *d_output.
 * 
 * @param d_input Pointer to device array [N].
 * @param d_output Pointer to device scalar to store the final maximum.
 * @param N Number of elements.
 * @param d_temp Optional pointer to device temporary storage for partial maxima.
 *               If nullptr, scratch memory is dynamically allocated.
 * @param stream CUDA stream (default: 0).
 */
void launchReduceMax(
    const float* d_input,
    float* d_output,
    size_t N,
    float* d_temp = nullptr,
    cudaStream_t stream = 0
);

/**
 * @brief Synchronous host convenience API for float32 max reduction.
 * 
 * Handles arbitrary N, including N = 0 (returns -INFINITY immediately without kernel launch).
 * 
 * @param d_input Device pointer to input array [N].
 * @param N Number of elements.
 * @param d_temp Optional pointer to device temporary storage.
 * @param stream CUDA stream (default: 0).
 * @return float The computed maximum value.
 */
float reduceMax(
    const float* d_input,
    size_t N,
    float* d_temp = nullptr,
    cudaStream_t stream = 0
);

/**
 * @brief Computes required temporary storage size in bytes for max reduction.
 */
size_t getReduceMaxTempStorageBytes(size_t N, int blockSize = 256);

/**
 * @brief Returns the number of blocks used in Pass 1 for max reduction.
 */
int getReduceMaxNumBlocks(size_t N, int blockSize = 256);

/**
 * @brief CPU reference implementation of max reduction with identical NaN and empty policies.
 * 
 * @param h_input Pointer to host array [N].
 * @param N Number of elements.
 * @return float Maximum element, or -INFINITY if N = 0.
 */
float reduceMaxCpuReference(
    const float* h_input,
    size_t N
);

/**
 * @brief Verifies GPU max reduction result against CPU reference.
 * 
 * Unlike sum reduction, max reduction has no rounding accumulation error.
 * Results must match exactly for standard floating-point values, or both be -INFINITY.
 * 
 * @param ref CPU reference maximum.
 * @param test GPU test maximum.
 * @return true if results match, false otherwise.
 */
bool verifyMaxReduction(
    float ref,
    float test
);

} // namespace gpu_roofline

