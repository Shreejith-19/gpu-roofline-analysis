#include "cuda_utils.cuh"
#include <iostream>
#include <vector>
#include <cassert>

__global__ void smokeTestKernel(int* data, int n, int multiplier) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        data[idx] = (idx + 1) * multiplier;
    }
}

int main() {
    try {
        std::cout << "[INFO] Starting CUDA Smoke Test..." << std::endl;

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        if (deviceCount == 0) {
            std::cerr << "[FAIL] No CUDA-capable devices detected." << std::endl;
            return 1;
        }

        // 1. Query & print device properties
        gpu_roofline::printDeviceProperties(0);

        // 2. Test memory allocation and kernel execution
        constexpr int N = 1 << 18; // 262,144 elements (~1 MB)
        constexpr size_t bytes = N * sizeof(int);
        constexpr int multiplier = 3;

        int* d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, bytes));

        gpu_roofline::GpuTimer timer;
        timer.start();

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        smokeTestKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N, multiplier);
        CUDA_CHECK_LAST_ERROR();

        timer.stop();
        float ms = timer.elapsedMillis();

        std::cout << "[INFO] Smoke test kernel executed in " << ms << " ms." << std::endl;

        // 3. Copy results back and verify correctness
        std::vector<int> h_data(N, 0);
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i) {
            int expected = (i + 1) * multiplier;
            if (h_data[i] != expected) {
                std::cerr << "[FAIL] Verification failed at index " << i 
                          << ": expected " << expected << ", got " << h_data[i] << std::endl;
                CUDA_CHECK(cudaFree(d_data));
                return 1;
            }
        }

        CUDA_CHECK(cudaFree(d_data));

        std::cout << "\n======================================================\n"
                  << "[SUCCESS] All checks passed: CUDA environment is operational!\n"
                  << "======================================================" << std::endl;

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[EXCEPTION] " << ex.what() << std::endl;
        return 1;
    }
}
