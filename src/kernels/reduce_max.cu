#include "reduce.cuh"
#include "cuda_utils.cuh"

#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <math_constants.h> // Provides CUDART_INF_F

namespace gpu_roofline {

/**
 * @brief Block-level max reduction kernel using grid-stride loops and shared-memory tree reduction.
 * 
 * Thread & Memory Mapping:
 *   - Each thread performs coalesced global-memory reads via grid-stride loops,
 *     accumulating local maxima in a private register initialized to -CUDART_INF_F.
 *   - Shared memory tree reduction uses sequential addressing (s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]))
 *     which completely eliminates bank conflicts.
 *   - Final 32 threads perform register-level warp shuffle down reduction (__shfl_down_sync).
 * 
 * NaN & Boundary Policies:
 *   - Identity Element: -INFINITY (-CUDART_INF_F), ensuring fmaxf(x, -inf) = x for all real x.
 *   - NaN Policy: Uses hardware fmaxf semantics, which treat NaN as a missing value (fmaxf(x, NaN) = x).
 *   - Out-of-bounds safety: Tail elements beyond N are never loaded; partial blocks pad with -CUDART_INF_F.
 */
template <unsigned int BLOCK_SIZE>
__global__ void reduceMaxBlockKernel(
    const float* __restrict__ input,
    float* __restrict__ partialMaxes,
    size_t N
) {
    extern __shared__ float s_data[];

    const unsigned int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    // 1. Grid-stride loop: Coalesced read and thread-local register max accumulation
    float localMax = -CUDART_INF_F;
    while (idx < N) {
        localMax = fmaxf(localMax, input[idx]);
        idx += stride;
    }

    // 2. Populate shared memory
    s_data[tid] = localMax;
    __syncthreads();

    // 3. Tree reduction in shared memory (sequential addressing: zero bank conflicts)
    for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }

    // 4. Final warp reduction (tid < 32)
    if (tid < 32) {
        float myVal = s_data[tid];
        if (BLOCK_SIZE >= 64) {
            myVal = fmaxf(myVal, s_data[tid + 32]);
        }

        // Warp shuffle down reduction (register-level, zero shared-memory latency)
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            myVal = fmaxf(myVal, __shfl_down_sync(0xffffffff, myVal, offset));
        }

        if (tid == 0) {
            partialMaxes[blockIdx.x] = myVal;
        }
    }
}

int getReduceMaxNumBlocks(size_t N, int blockSize) {
    if (N == 0) return 0;
    size_t neededBlocks = (N + blockSize - 1) / blockSize;
    constexpr int MAX_BLOCKS = 256;
    return static_cast<int>(std::min(neededBlocks, static_cast<size_t>(MAX_BLOCKS)));
}

size_t getReduceMaxTempStorageBytes(size_t N, int blockSize) {
    int numBlocks = getReduceMaxNumBlocks(N, blockSize);
    return static_cast<size_t>(numBlocks) * sizeof(float);
}

void launchReduceMax(
    const float* d_input,
    float* d_output,
    size_t N,
    float* d_temp,
    cudaStream_t stream
) {
    if (d_output == nullptr) {
        throw std::invalid_argument("d_output pointer cannot be null.");
    }

    // Explicitly handle empty input N = 0 without launching kernels
    if (N == 0) {
        float negInf = -std::numeric_limits<float>::infinity();
        CUDA_CHECK(cudaMemcpyAsync(d_output, &negInf, sizeof(float), cudaMemcpyHostToDevice, stream));
        return;
    }

    if (d_input == nullptr) {
        throw std::invalid_argument("d_input pointer cannot be null when N > 0.");
    }

    constexpr int BLOCK_SIZE = 256;
    int numBlocks = getReduceMaxNumBlocks(N, BLOCK_SIZE);

    bool allocatedTemp = false;
    float* tempBuffer = d_temp;

    // Allocate temporary scratchpad buffer if caller did not supply one
    if (tempBuffer == nullptr && numBlocks > 1) {
        CUDA_CHECK(cudaMallocAsync(&tempBuffer, numBlocks * sizeof(float), stream));
        allocatedTemp = true;
    }

    size_t smemBytes = BLOCK_SIZE * sizeof(float);

    if (numBlocks == 1) {
        // Direct reduction to final output
        reduceMaxBlockKernel<BLOCK_SIZE><<<1, BLOCK_SIZE, smemBytes, stream>>>(d_input, d_output, N);
    } else {
        // Pass 1: Reduce N global elements into numBlocks partial maxima
        reduceMaxBlockKernel<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, smemBytes, stream>>>(d_input, tempBuffer, N);

        // Pass 2: Reduce numBlocks partial maxima into single scalar d_output
        reduceMaxBlockKernel<BLOCK_SIZE><<<1, BLOCK_SIZE, smemBytes, stream>>>(tempBuffer, d_output, numBlocks);
    }

    CUDA_CHECK_LAST_ERROR();

    if (allocatedTemp) {
        CUDA_CHECK(cudaFreeAsync(tempBuffer, stream));
    }
}

float reduceMax(
    const float* d_input,
    size_t N,
    float* d_temp,
    cudaStream_t stream
) {
    // Explicit N = 0 host check
    if (N == 0) {
        return -std::numeric_limits<float>::infinity();
    }

    float* d_output = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_output, sizeof(float), stream));

    launchReduceMax(d_input, d_output, N, d_temp, stream);

    float h_result = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_output, stream));

    return h_result;
}

float reduceMaxCpuReference(const float* h_input, size_t N) {
    if (N == 0) {
        return -std::numeric_limits<float>::infinity();
    }

    float maxVal = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < N; ++i) {
        float val = h_input[i];
        if (!std::isnan(val)) {
            if (val > maxVal) {
                maxVal = val;
            }
        }
    }
    return maxVal;
}

bool verifyMaxReduction(float ref, float test) {
    // Handle infinite results (e.g. empty or all -inf/NaNs)
    if (std::isinf(ref) && std::isinf(test)) {
        return (ref < 0 && test < 0) || (ref > 0 && test > 0);
    }

    // Unlike sum reduction, max reduction has no rounding accumulation error.
    // Real float values must match exactly.
    if (ref != test) {
        float diff = std::fabs(ref - test);
        std::cerr << "  [VERIFY FAILED] CPU ref=" << ref << ", GPU=" << test 
                  << ", diff=" << diff << "\n";
        return false;
    }
    return true;
}

} // namespace gpu_roofline
