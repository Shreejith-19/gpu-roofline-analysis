#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iomanip>

/**
 * @brief Macro for checking CUDA Runtime API calls.
 */
#define CUDA_CHECK(ans) do { ::gpu_roofline::cudaAssert((ans), __FILE__, __LINE__, #ans); } while (false)

/**
 * @brief Macro for checking asynchronous kernel launch errors.
 */
#define CUDA_CHECK_LAST_ERROR() do { \
    cudaError_t const err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::stringstream ss; \
        ss << "CUDA Kernel Launch Error [" << cudaGetErrorName(err) << "]: " \
           << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__; \
        std::cerr << "[ERROR] " << ss.str() << std::endl; \
        throw std::runtime_error(ss.str()); \
    } \
} while (false)

namespace gpu_roofline {

/**
 * @brief Handles CUDA runtime errors by printing diagnostic info and throwing std::runtime_error.
 */
inline void cudaAssert(cudaError_t code, const char* file, int line, const char* expression, bool abort = true) {
    if (code != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA Error [" << cudaGetErrorName(code) << "] (" << code << "): "
           << cudaGetErrorString(code)
           << " in expression '" << expression << "' at "
           << file << ":" << line;
        std::cerr << "\n================== CUDA ERROR ==================\n"
                  << ss.str() << "\n"
                  << "================================================\n" << std::endl;
        if (abort) {
            throw std::runtime_error(ss.str());
        }
    }
}

/**
 * @brief RAII CUDA Event Timer for benchmarking GPU kernel execution time in milliseconds.
 */
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Disable copy
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    /**
     * @brief Synchronizes and returns elapsed time in milliseconds.
     */
    float elapsedMillis() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

/**
 * @brief Computes theoretical peak memory bandwidth in GB/s.
 * Peak Bandwidth (GB/s) = (Memory Clock in kHz * 1000 * 2 (DDR) * (Bus Width / 8 bits)) / 10^9
 */
inline double getTheoreticalPeakBandwidthGBs(int deviceId) {
    int memoryClockKhz = 0;
    int busWidthBits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memoryClockKhz, cudaDevAttrMemoryClockRate, deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, deviceId));

    double memoryClockHz = static_cast<double>(memoryClockKhz) * 1e3; // kHz -> Hz
    double busWidthBytes = static_cast<double>(busWidthBits) / 8.0;
    // DDR multiplier is 2.0
    return (2.0 * memoryClockHz * busWidthBytes) / 1e9;
}

/**
 * @brief Prints hardware specification and compute characteristics of the targeted device.
 */
inline void printDeviceProperties(int deviceId = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    int memoryClockKhz = 0;
    int busWidthBits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memoryClockKhz, cudaDevAttrMemoryClockRate, deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, deviceId));

    double totalMemGB = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
    double peakBwGBs = getTheoreticalPeakBandwidthGBs(deviceId);

    std::cout << "================== GPU Device Info ==================\n"
              << " Device ID               : " << deviceId << "\n"
              << " Device Name             : " << prop.name << "\n"
              << " Compute Capability      : " << prop.major << "." << prop.minor << "\n"
              << " Streaming Multiprocessors: " << prop.multiProcessorCount << "\n"
              << " Max Threads Per Block   : " << prop.maxThreadsPerBlock << "\n"
              << " Warp Size               : " << prop.warpSize << "\n"
              << " Total Global VRAM       : " << std::fixed << std::setprecision(2) << totalMemGB << " GB (" 
              << prop.totalGlobalMem / (1024 * 1024) << " MiB)\n"
              << " Memory Bus Width        : " << busWidthBits << " bits\n"
              << " Memory Clock Rate       : " << memoryClockKhz / 1000 << " MHz\n"
              << " Theoretical Peak Bandwidth: " << std::fixed << std::setprecision(2) << peakBwGBs << " GB/s\n"
              << "=====================================================" << std::endl;
}

} // namespace gpu_roofline
