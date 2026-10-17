#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace gpu_roofline {

/**
 * @brief Benchmark result entry for exporting to CSV.
 */
struct BenchmarkRecord {
    std::string kernelName;
    std::string memoryType;        // e.g. "Global", "Shared", "Unified"
    size_t problemSize;            // Number of elements
    double bytesTransferred;       // Total theoretical bytes transferred
    double flosExecuted;           // Total floating-point operations
    double elapsedMillis;          // Kernel duration in ms
    double achievedBandwidthGBs;   // (Bytes / 1e9) / (time_sec)
    double achievedGflops;         // (FLOPs / 1e9) / (time_sec)
    double operationalIntensity;   // FLOPs / Byte
};

/**
 * @brief CSV Writer utility for exporting benchmark records to the results folder.
 */
class CsvWriter {
public:
    explicit CsvWriter(const std::string& filepath)
        : filepath_(filepath) {}

    bool writeHeader() {
        std::ofstream file(filepath_, std::ios::out);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Could not open file for writing: " << filepath_ << std::endl;
            return false;
        }

        file << "KernelName,MemoryType,ProblemSize,BytesTransferred,FlopsExecuted,"
             << "ElapsedMillis,AchievedBandwidthGBs,AchievedGflops,OperationalIntensity\n";
        return true;
    }

    bool appendRecord(const BenchmarkRecord& rec) {
        std::ofstream file(filepath_, std::ios::app);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Could not open file for appending: " << filepath_ << std::endl;
            return false;
        }

        file << rec.kernelName << ","
             << rec.memoryType << ","
             << rec.problemSize << ","
             << std::fixed << std::setprecision(0) << rec.bytesTransferred << ","
             << std::fixed << std::setprecision(0) << rec.flosExecuted << ","
             << std::fixed << std::setprecision(4) << rec.elapsedMillis << ","
             << std::fixed << std::setprecision(4) << rec.achievedBandwidthGBs << ","
             << std::fixed << std::setprecision(4) << rec.achievedGflops << ","
             << std::fixed << std::setprecision(6) << rec.operationalIntensity << "\n";
        return true;
    }

private:
    std::string filepath_;
};

} // namespace gpu_roofline
