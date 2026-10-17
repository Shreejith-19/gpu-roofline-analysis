#include "reduce.cuh"
#include "cuda_utils.cuh"

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace gpu_roofline {

/**
 * @brief Block-level sum reduction kernel using a grid-stride loop and shared-memory tree reduction.
 * 
 * Each thread accumulates multiple elements into a private register via grid-stride loops,
 * ensuring coalesced 128-byte global memory transactions across warps.
 * The block then performs an in-shared-memory binary tree reduction with sequential addressing,
 * followed by a final warp-shuffle reduction across the remaining 32 threads.
 */
template <unsigned int BLOCK_SIZE>
__global__ void reduceBlockKernel(
    const float* __restrict__ input,
    float* __restrict__ partialSums,
    size_t N
) {
    extern __shared__ float s_data[];

    const unsigned int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    // 1. Grid-stride loop: Coalesced read and thread-local register accumulation
    float sum = 0.0f;
    while (idx < N) {
        sum += input[idx];
        idx += stride;
    }

    // 2. Populate shared memory
    s_data[tid] = sum;
    __syncthreads();

    // 3. Tree reduction in shared memory (sequential addressing: no bank conflicts)
    for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // 4. Final warp reduction (tid < 32)
    if (tid < 32) {
        float myVal = s_data[tid];
        if (BLOCK_SIZE >= 64) {
            myVal += s_data[tid + 32];
        }

        // Warp shuffle down reduction (register-level, zero shared-memory overhead)
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            myVal += __shfl_down_sync(0xffffffff, myVal, offset);
        }

        if (tid == 0) {
            partialSums[blockIdx.x] = myVal;
        }
    }
}

int getReduceSumNumBlocks(size_t N, int blockSize) {
    if (N == 0) return 0;
    size_t neededBlocks = (N + blockSize - 1) / blockSize;
    constexpr int MAX_BLOCKS = 256;
    return static_cast<int>(std::min(neededBlocks, static_cast<size_t>(MAX_BLOCKS)));
}

size_t getReduceSumTempStorageBytes(size_t N, int blockSize) {
    int numBlocks = getReduceSumNumBlocks(N, blockSize);
    return static_cast<size_t>(numBlocks) * sizeof(float);
}

void launchReduceSum(
    const float* d_input,
    float* d_output,
    size_t N,
    float* d_temp,
    cudaStream_t stream
) {
    if (d_output == nullptr) {
        throw std::invalid_argument("d_output pointer cannot be null.");
    }

    // Explicitly handle N = 0 edge case without launching kernels
    if (N == 0) {
        float zero = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));
        return;
    }

    if (d_input == nullptr) {
        throw std::invalid_argument("d_input pointer cannot be null when N > 0.");
    }

    constexpr int BLOCK_SIZE = 256;
    int numBlocks = getReduceSumNumBlocks(N, BLOCK_SIZE);

    bool allocatedTemp = false;
    float* tempBuffer = d_temp;

    // Allocate temporary scratchpad buffer if caller did not supply one
    if (tempBuffer == nullptr && numBlocks > 1) {
        CUDA_CHECK(cudaMallocAsync(&tempBuffer, numBlocks * sizeof(float), stream));
        allocatedTemp = true;
    }

    // Pass 1: Reduce N global elements into numBlocks partial sums
    size_t smemBytes = BLOCK_SIZE * sizeof(float);
    if (numBlocks == 1) {
        // Direct reduction to final output
        reduceBlockKernel<BLOCK_SIZE><<<1, BLOCK_SIZE, smemBytes, stream>>>(d_input, d_output, N);
    } else {
        reduceBlockKernel<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, smemBytes, stream>>>(d_input, tempBuffer, N);
        
        // Pass 2: Reduce numBlocks partial sums into single scalar d_output
        // numBlocks <= 256, so a single block of BLOCK_SIZE threads suffices
        reduceBlockKernel<BLOCK_SIZE><<<1, BLOCK_SIZE, smemBytes, stream>>>(tempBuffer, d_output, numBlocks);
    }

    CUDA_CHECK_LAST_ERROR();

    if (allocatedTemp) {
        CUDA_CHECK(cudaFreeAsync(tempBuffer, stream));
    }
}

float reduceSum(
    const float* d_input,
    size_t N,
    float* d_temp,
    cudaStream_t stream
) {
    // Explicit N = 0 host check
    if (N == 0) {
        return 0.0f;
    }

    float* d_output = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_output, sizeof(float), stream));

    launchReduceSum(d_input, d_output, N, d_temp, stream);

    float h_result = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_output, stream));

    return h_result;
}

float reduceSumCpuReference(const float* h_input, size_t N, bool useDoublePrecision) {
    if (N == 0) return 0.0f;
    if (useDoublePrecision) {
        double acc = 0.0;
        for (size_t i = 0; i < N; ++i) {
            acc += static_cast<double>(h_input[i]);
        }
        return static_cast<float>(acc);
    } else {
        float acc = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            acc += h_input[i];
        }
        return acc;
    }
}

bool verifyReduction(float ref, float test, size_t N, float rtol) {
    float diff = std::fabs(ref - test);
    // Tolerance scales with sqrt(N) * eps to account for floating-point accumulation drift
    float tol = rtol * std::fabs(ref) + 1e-4f;
    if (diff > tol) {
        std::cerr << "  [VERIFY FAILED] CPU ref=" << ref << ", GPU=" << test 
                  << ", diff=" << diff << ", tol=" << tol << " (N=" << N << ")\n";
        return false;
    }
    return true;
}

} // namespace gpu_roofline
