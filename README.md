# GPU Performance Analysis and Roofline Analysis

[![CUDA](https://img.shields.io/badge/CUDA-12.x%20%7C%2013.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/CMake-3.24%2B-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/GPU-Ampere%20(SM%2086)-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com)

A high-performance CUDA C++ and Python framework for empirical GPU performance modeling, hardware counter extraction, and Roofline analysis on modern NVIDIA architectures. 

This repository provides custom, zero-library implementations of compute-bound and memory-bound CUDA kernels (Matrix Multiplication and Parallel Reductions), a complete benchmarking and profiling pipeline using NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`), and automated scripts generating publication-grade empirical Roofline models and comparative performance visualizations.

---

## Table of Contents
1. [Architectural Overview](#architectural-overview)
2. [Target Hardware Specifications](#target-hardware-specifications)
3. [Repository Structure](#repository-structure)
4. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
5. [Building the Project](#building-the-project)
6. [How to Run the Project](#how-to-run-the-project)
   - [1. Verification & Unit Tests](#1-verification--unit-tests)
   - [2. Benchmark Sweeps](#2-benchmark-sweeps)
   - [3. Visual Plot Generation](#3-visual-plot-generation)
   - [4. Microarchitectural Profiling Workflow](#4-microarchitectural-profiling-workflow)
7. [Theoretical Foundations & Metric Formulations](#theoretical-foundations--metric-formulations)
8. [Empirical Benchmark Results](#empirical-benchmark-results)
9. [Nsight Profiling & Microarchitectural Insights](#nsight-profiling--microarchitectural-insights)

---

## Architectural Overview

This framework characterizes GPU hardware capability by comparing measured kernel performance against theoretical physical ceilings via the **Empirical Roofline Model**:

```
Attainable Performance (GFLOP/s)
  ^
  |                     Peak FP32 Compute Ceiling (7,200 GFLOP/s)
  |                  +----------------------------------------------+ (Compute-Bound)
  |                 /
  |                /
  |               /  <- Ridge Point: 37.5 FLOP/B
  |              /
  |             /
  |            /  Peak DRAM Bandwidth Slope (192.0 GB/s)
  |           /   (Memory-Bound)
  |          /
  +---------+------------------------------------------------------->
  0.1      1.0                   10.0               100.0          1000.0
                       Operational Intensity (FLOPs / Byte)
```

### Implemented CUDA Kernels
- **Baseline Naive GEMM** ([`src/kernels/matmul_naive.cu`](file:///c:/Users/shree/Documents/CUDA/Projects/gpu-roofline-analysis/src/kernels/matmul_naive.cu)):
  Standard row-major FP32 matrix multiplication ($C = A \times B$). One thread per output element computing inner products through global memory loads only. Serves as the un-optimized baseline illustrating memory bandwidth bottlenecks on high-arithmetic-intensity algorithms.
- **16×16 Tiled Shared-Memory GEMM** ([`src/kernels/matmul_tiled.cu`](file:///c:/Users/shree/Documents/CUDA/Projects/gpu-roofline-analysis/src/kernels/matmul_tiled.cu)):
  Cooperative global-to-shared memory tile staging using 2D thread blocks ($16 \times 16$). Reduces global DRAM transactions by a factor of $16\times$ via data reuse in on-chip SRAM, accelerating throughput by up to **$+41\%$**.
- **Parallel Sum Reduction** ([`src/kernels/reduce_sum.cu`](file:///c:/Users/shree/Documents/CUDA/Projects/gpu-roofline-analysis/src/kernels/reduce_sum.cu)):
  Two-stage parallel reduction utilizing grid-stride coalesced loops, warp-level shuffle registers (`__shfl_down_sync`), and shared-memory block-level reduction trees. Achieves **$179.03\text{ GB/s}$ ($93.2\%$ of physical DRAM peak)**.
- **Parallel Max Reduction** ([`src/kernels/reduce_max.cu`](file:///c:/Users/shree/Documents/CUDA/Projects/gpu-roofline-analysis/src/kernels/reduce_max.cu)):
  High-throughput streaming max reduction with strict IEEE-754 semantics: negative values, identical duplicates, zero-element edge cases, and consistent NaN handling. Sustains **$178.93\text{ GB/s}$**.

---

## Target Hardware Specifications

All benchmark and profiling baselines are tailored to and measured on the following hardware platform:

| Specification | Hardware Parameter | Architectural Details |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 3050 Laptop GPU | GA107 silicon, Ampere Architecture |
| **CUDA Compute Capability** | `sm_86` | Dual-path FP32/INT32 execution units |
| **Streaming Multiprocessors (SMs)** | 16 SMs | 128 FP32 ALUs per SM (2,048 FP32 cores total) |
| **Max Warps per SM** | 48 warps (1,536 active threads/SM) | 24,576 maximum resident threads across GPU |
| **Theoretical Peak FP32 Compute** | **7,200 GFLOP/s (7.2 TFLOP/s)** | $2048 \text{ cores} \times 1757\text{ MHz} \times 2\text{ FLOP/clock}$ |
| **Global Memory Capacity** | 4.00 GB GDDR6 | Dedicated VRAM (3,980 MiB usable) |
| **Memory Bus Width** | 128 bits | 4 memory controllers $\times$ 32-bit channels |
| **Memory Clock Rate** | 6001 MHz (12 Gbps effective) | $128\text{ bits} \times 6001\text{ MHz} \times 2 / 8 = \mathbf{192.03\text{ GB/s}}$ |
| **L2 Cache Size** | 2,048 KB (2.0 MB) | High-speed on-chip crossbar cache |
| **Shared Memory / L1 per SM** | Configurable up to 100 KB | 128 KB unified data cache / shared memory |
| **Roofline Ridge Point** | **37.5 FLOP/Byte** | $7,200\text{ GFLOP/s} / 192.03\text{ GB/s}$ |

---

## Repository Structure

```text
gpu-roofline-analysis/
├── CMakeLists.txt                  # Top-level CMake build configuration
├── README.md                       # Comprehensive project documentation
├── .gitignore                      # Git ignore patterns for build artifacts and profiles
├── include/                        # Public C++ and CUDA headers
│   ├── cuda_utils.cuh              # CUDA error macros, GpuTimer (CUDA events), device query
│   ├── common.hpp                  # Benchmark structures, timing utilities, CSV exporter
│   ├── matmul.cuh                  # Function signatures for Naive & Tiled GEMM kernels
│   └── reduce.cuh                  # Host APIs and device declarations for Sum & Max Reductions
├── src/
│   ├── kernels/                    # Production CUDA kernel implementations
│   │   ├── matmul_naive.cu         # Baseline global-memory matrix multiplication
│   │   ├── matmul_tiled.cu         # 16x16 shared-memory cached matrix multiplication
│   │   ├── reduce_sum.cu           # Grid-stride + warp shuffle float32 sum reduction
│   │   └── reduce_max.cu           # Hierarchical partial-max + warp shuffle max reduction
│   └── benchmarks/                 # Standalone benchmarking drivers
│       ├── benchmark_matmul.cu     # Parameter sweep driver for Naive vs Tiled GEMM
│       ├── benchmark_reduce.cu     # Scaling benchmark driver for Sum Reduction (up to 64M)
│       └── benchmark_reduce_max.cu # Scaling benchmark driver for Max Reduction (up to 64M)
├── tests/                          # Integration and correctness validation suite
│   ├── CMakeLists.txt              # CTest target declarations
│   ├── smoke_test.cu               # Device capability and memory transfer sanity test
│   ├── test_matmul_naive.cu        # Naive GEMM unit tests against CPU reference
│   ├── test_matmul_tiled.cu        # Tiled GEMM unit tests vs Naive and CPU
│   ├── test_reduce_sum.cu          # Sum reduction unit tests (tolerances, edge cases)
│   └── test_reduce_max.cu          # Max reduction unit tests (NaNs, negatives, primes)
├── scripts/                        # Automation, profiling, and visualization scripts
│   ├── plot_roofline.py            # Generates publication-ready Empirical Roofline models
│   ├── plot_comparisons.py         # Generates GEMM and Reduction comparative metric charts
│   ├── profile_kernels.py          # Nsight Systems & Nsight Compute automation harness
│   └── run_benchmarks.py           # Benchmark orchestration and build runner
└── results/                        # Benchmark output artifacts and profiling data
    ├── benchmark_results.csv       # Measured kernel metrics (GFLOP/s, GB/s, latency, OI)
    ├── plots/                      # High-resolution raster (PNG) and vector (SVG) charts
    │   ├── roofline_model.png      # Empirical Roofline model plot
    │   ├── matmul_comparison.png   # Naive vs Tiled GEMM comparison chart
    │   └── reduction_comparison.png# Sum vs Max Reduction comparison chart
    └── profiles/                   # Nsight timeline (.nsys-rep) and SQLite trace exports
```

---

## Prerequisites & Environment Setup

### 1. Toolchain Requirements
- **Operating System**: Windows 10/11 (64-bit) or Linux (Ubuntu 20.04+)
- **NVIDIA CUDA Toolkit**: v12.x or v13.x (`nvcc` compiler)
- **C++ Compiler**: Microsoft Visual C++ (MSVC `cl.exe` 19.30+ / Visual Studio 2022) or GCC 11+
- **Build System**: CMake $\ge$ 3.24
- **Python Runtime**: Python 3.10+ with `numpy` and `matplotlib`
- **Profilers (Optional)**: NVIDIA Nsight Compute (`ncu`) $\ge$ 2026.1 and NVIDIA Nsight Systems (`nsys`) $\ge$ 2026.1

### 2. Verify Your Environment
Run the following commands in PowerShell to verify tool availability:

```powershell
# Verify CUDA Compiler
nvcc --version

# Verify CMake
cmake --version

# Verify Visual Studio C++ Compiler
cl

# Verify Python & Visualization Packages
python --version
python -c "import numpy, matplotlib; print('Python packages verified!')"

# Verify NVIDIA Profilers
ncu --version
nsys --version
```

---

## Building the Project

The project uses modern CMake with native CUDA language support (`enable_language(CUDA)`).

### Step 1: Configure CMake
Configure the build system. The architecture is configured via `-DCMAKE_CUDA_ARCHITECTURES`. For the RTX 3050 Laptop GPU, use `86` (Ampere `sm_86`):

```powershell
# Configure for Ampere SM 86 (RTX 3050 / RTX 3060)
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=86
```
*(Optional: Use `-DCMAKE_CUDA_ARCHITECTURES=native` to automatically detect your host GPU architecture).*

### Step 2: Compile in Release Mode
Build all kernel libraries, benchmark executables, and test targets:

```powershell
cmake --build build --config Release
```

Upon completion, all binaries will be generated in `build/Release/` (benchmarks) and `build/tests/Release/` (unit tests).

---

## How to Run the Project

### 1. Verification & Unit Tests
Run the entire validation test suite using `ctest`. This tests square, rectangular, irregular, prime, and edge-case inputs against CPU reference implementations:

```powershell
ctest --test-dir build -C Release --output-on-failure
```
*Expected output: `100% tests passed, 0 tests failed out of 5`.*

---

### 2. Benchmark Sweeps
Run the benchmark executables to measure latency, throughput, and memory bandwidth across matrix dimensions and buffer sizes. All benchmarks automatically populate `results/benchmark_results.csv`:

#### A. Matrix Multiplication Benchmark (Naive vs. 16×16 Tiled)
Benchmarks 11 matrix configurations (from $64 \times 64$ to $2048 \times 2048$, plus tall-skinny and irregular matrices):
```powershell
.\build\Release\benchmark_matmul.exe
```

#### B. Parallel Sum Reduction Benchmark
Sweeps floating-point buffer sizes from $64\text{K}$ elements ($256\text{ KB}$) up to $64\text{M}$ elements ($256\text{ MB}$):
```powershell
.\build\Release\benchmark_reduce.exe
```

#### C. Parallel Max Reduction Benchmark
Benchmarks max-reduction comparison throughput and bandwidth saturation up to $64\text{M}$ elements:
```powershell
.\build\Release\benchmark_reduce_max.exe
```

---

### 3. Visual Plot Generation

Generate high-resolution (300 DPI) raster (`.png`) and publication-quality vector (`.svg`) plots:

#### A. Empirical Roofline Model
Maps every tested kernel configuration onto the theoretical hardware limits of the RTX 3050:
```powershell
python scripts\plot_roofline.py
```
*Output: [`results/plots/roofline_model.png`](file:///c:/Users/shree/Documents/CUDA/Projects/gpu-roofline-analysis/results/plots/roofline_model.png) and [`roofline_model.svg`](file:///c:/Users/shree/Documents/CUDA/Projects/gpu-roofline-analysis/results/plots/roofline_model.svg)*

#### B. Kernel Comparative Evaluations (with Formal Metrics)
Generates dedicated multi-panel comparison charts complete with formal mathematical formulations and metric cards:
```powershell
python scripts\plot_comparisons.py
```
*Outputs:*
- **GEMM Comparison**: [`results/plots/matmul_comparison.png`](file:///c:/Users/shree/Documents/CUDA/Projects/gpu-roofline-analysis/results/plots/matmul_comparison.png) (Throughput GFLOP/s, Speedup multipliers, Operational Intensity formulas).
- **Reduction Comparison**: [`results/plots/reduction_comparison.png`](file:///c:/Users/shree/Documents/CUDA/Projects/gpu-roofline-analysis/results/plots/reduction_comparison.png) (Effective streaming GB/s, G-Elements/s processing rates, DRAM saturation).

---

### 4. Microarchitectural Profiling Workflow

We provide an automated CLI profiling script ([`scripts/profile_kernels.py`](file:///c:/Users/shree/Documents/CUDA/Projects/gpu-roofline-analysis/scripts/profile_kernels.py)) supporting **Nsight Systems** (timeline tracing, CPU-side kernel launch overheads) and **Nsight Compute** (SM throughput, occupancy, register pressure, memory traffic, warp stalls, and Roofline charts).

#### Run Timeline Profiling via Nsight Systems:
```powershell
# Profile all four kernels
python scripts\profile_kernels.py --tool nsys --kernel all

# Profile a specific kernel
python scripts\profile_kernels.py --tool nsys --kernel matmul_tiled
```

#### Run Deep Hardware Metric Profiling via Nsight Compute:
```powershell
# Profile verified core hardware metrics (registers, occupancy, stalls, DRAM bandwidth)
python scripts\profile_kernels.py --tool ncu --kernel matmul_tiled --ncu-mode metrics

# Generate full Nsight Compute Roofline Section (.ncu-rep)
python scripts\profile_kernels.py --tool ncu --kernel reduce_sum --ncu-mode roofline
```

---

## Theoretical Foundations & Metric Formulations

### 1. The Roofline Model
The Roofline model bounds the attainable execution throughput $P$ as a function of the kernel's **Operational Intensity** $I$:

$$P = \min\left(P_{\text{peak}},\, I \times \text{BW}_{\text{peak}}\right)$$

Where:
- $P_{\text{peak}}$: Theoretical maximum compute throughput (**$7,200\text{ GFLOP/s}$** on RTX 3050).
- $\text{BW}_{\text{peak}}$: Theoretical maximum memory bandwidth (**$192.03\text{ GB/s}$** on RTX 3050).
- **Ridge Point ($I_{\text{ridge}}$)**: The inflection intensity where the memory-bound slope meets the compute-bound ceiling:
  $$I_{\text{ridge}} = \frac{P_{\text{peak}}}{\text{BW}_{\text{peak}}} = \frac{7200\text{ GFLOP/s}}{192.03\text{ GB/s}} = \mathbf{37.50\text{ FLOP/Byte}}$$

### 2. Performance Metrics Definition

| Metric | Mathematical Formulation | Architectural Significance |
| :--- | :--- | :--- |
| **GEMM FLOPs** | $\text{FLOPs} = 2 \times M \times N \times K$ | 1 multiply + 1 addition (Fused Multiply-Add) per inner-loop iteration for $A[M,K] \times B[K,N]$. |
| **Compute Throughput** | $\text{GFLOP/s} = \frac{\text{FLOPs}}{\text{Time (s)} \times 10^9}$ | Rate of floating-point arithmetic sustained across all active SM pipelines. |
| **Minimum DRAM Traffic** | $\text{Bytes} = 4 \times (MK + KN + MN)$ | Compulsory data volume assuming a single read of $A$ and $B$, single write of $C$ (FP32 = 4 bytes). |
| **Operational Intensity (GEMM)** | $\text{OI} = \frac{2MNK}{4(MK + KN + MN)}$ | Ratio of computation to memory traffic. Increases proportionally with matrix dimension ($M, N, K$). |
| **Effective Bandwidth (Reduction)** | $\text{BW}_{\text{eff}} = \frac{N \times 4\text{ Bytes}}{\text{Time (s)} \times 10^9}$ | Rate of data streamed from GDDR6 across grid-stride loops. |
| **DRAM Bus Saturation** | $\text{Saturation} = \frac{\text{BW}_{\text{eff}}}{\text{BW}_{\text{peak}}} \times 100\%$ | Hardware efficiency relative to physical 128-bit bus capacity ($192.03\text{ GB/s}$). |
| **Speedup Factor** | $\text{Speedup} = \frac{T_{\text{naive}}}{T_{\text{tiled}}}$ | Latency ratio; $>1.0\times$ quantifies gain achieved via shared-memory tile staging. |

---

## Empirical Benchmark Results

Measured empirical baselines on NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.4, Driver 591.44):

### Matrix Multiplication: Naive vs. 16×16 Tiled GEMM
| Matrix Dimensions ($M \times N \times K$) | Naive Latency | Tiled Latency | Naive GFLOP/s | Tiled GFLOP/s | Speedup | Algorithmic OI | Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$64 \times 64 \times 64$** | $0.0085\text{ ms}$ | $0.0092\text{ ms}$ | $61.4\text{ GFLOP/s}$ | $56.9\text{ GFLOP/s}$ | $0.92\times$ | $10.67\text{ FLOP/B}$ | Memory-Bound |
| **$128 \times 128 \times 128$** | $0.0166\text{ ms}$ | $0.0129\text{ ms}$ | $252.8\text{ GFLOP/s}$ | $325.3\text{ GFLOP/s}$ | **$1.29\times$** | $21.33\text{ FLOP/B}$ | Memory-Bound |
| **$256 \times 256 \times 256$** | $0.0974\text{ ms}$ | $0.0732\text{ ms}$ | $344.6\text{ GFLOP/s}$ | $458.3\text{ GFLOP/s}$ | **$1.33\times$** | $42.67\text{ FLOP/B}$ | Compute-Bound |
| **$512 \times 512 \times 512$** | $0.7268\text{ ms}$ | $0.5493\text{ ms}$ | $369.3\text{ GFLOP/s}$ | $488.7\text{ GFLOP/s}$ | **$1.32\times$** | $85.33\text{ FLOP/B}$ | Compute-Bound |
| **$1024 \times 1024 \times 1024$** | $5.9091\text{ ms}$ | $4.3244\text{ ms}$ | $363.4\text{ GFLOP/s}$ | $496.6\text{ GFLOP/s}$ | **$1.37\times$** | $170.67\text{ FLOP/B}$ | Compute-Bound |
| **$2048 \times 2048 \times 2048$** | $43.3326\text{ ms}$| $30.6893\text{ ms}$| $396.5\text{ GFLOP/s}$ | **$559.8\text{ GFLOP/s}$** | **$1.41\times$** | $341.33\text{ FLOP/B}$ | Compute-Bound |
| **$67 \times 113 \times 97$ (Irreg)**| $0.0298\text{ ms}$ | $0.0304\text{ ms}$ | $49.2\text{ GFLOP/s}$  | $48.4\text{ GFLOP/s}$  | $0.98\times$ | $14.67\text{ FLOP/B}$ | Memory-Bound |
| **$131 \times 73 \times 59$ (Irreg)**| $0.0131\text{ ms}$ | $0.0094\text{ ms}$ | $86.1\text{ GFLOP/s}$  | $119.8\text{ GFLOP/s}$ | **$1.39\times$** | $13.06\text{ FLOP/B}$ | Memory-Bound |
| **$512 \times 64 \times 128$ (Tall)** | $0.0278\text{ ms}$ | $0.0210\text{ ms}$ | $301.7\text{ GFLOP/s}$ | $398.6\text{ GFLOP/s}$ | **$1.32\times$** | $19.69\text{ FLOP/B}$ | Memory-Bound |
| **$64 \times 512 \times 256$ (Wide)** | $0.0520\text{ ms}$ | $0.0391\text{ ms}$ | $322.5\text{ GFLOP/s}$ | $429.0\text{ GFLOP/s}$ | **$1.33\times$** | $23.27\text{ FLOP/B}$ | Memory-Bound |

### Parallel Reductions: Sum vs. Max Reduction
| Problem Size ($N$ Elements) | Data Volume (MB) | Sum Bandwidth | Max Bandwidth | Sum G-Elem/s | Max G-Elem/s | DRAM Saturation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$65,536$ ($64\text{K}$)** | $0.25\text{ MB}$ | $15.94\text{ GB/s}$ | $7.35\text{ GB/s}$ | $3.99\text{ GElem/s}$ | $1.84\text{ GElem/s}$ | $8.3\%$ |
| **$262,144$ ($256\text{K}$)** | $1.00\text{ MB}$ | $56.59\text{ GB/s}$ | $35.73\text{ GB/s}$ | $14.15\text{ GElem/s}$| $8.93\text{ GElem/s}$ | $29.5\%$ |
| **$1,048,576$ ($1\text{M}$)** | $4.00\text{ MB}$ | $134.80\text{ GB/s}$| $134.44\text{ GB/s}$| $33.70\text{ GElem/s}$| $33.61\text{ GElem/s}$| $70.2\%$ |
| **$4,194,304$ ($4\text{M}$)** | $16.00\text{ MB}$ | $164.97\text{ GB/s}$| $163.91\text{ GB/s}$| $41.24\text{ GElem/s}$| $40.98\text{ GElem/s}$| $85.9\%$ |
| **$10,000,007$ (Irreg 10M)** | $38.15\text{ MB}$ | $172.92\text{ GB/s}$| $170.50\text{ GB/s}$| $43.23\text{ GElem/s}$| $42.63\text{ GElem/s}$| $90.1\%$ |
| **$16,777,216$ ($16\text{M}$)**| $64.00\text{ MB}$ | $175.66\text{ GB/s}$| $175.34\text{ GB/s}$| $43.91\text{ GElem/s}$| $43.84\text{ GElem/s}$| $91.5\%$ |
| **$67,108,864$ ($64\text{M}$)**| $256.00\text{ MB}$| **$179.03\text{ GB/s}$**| **$178.93\text{ GB/s}$**| **$44.76\text{ GElem/s}$**| **$44.73\text{ GElem/s}$**| **$93.2\%$** |

---

## Nsight Profiling & Microarchitectural Insights

### Verified Metric Table (Nsight Compute 2026.3.1.0 on Ampere GA107)

```text
├── Resource Allocation
│   ├── launch__registers_per_thread              # Registers allocated per thread (Max 255)
│   └── launch__shared_mem_per_block_allocated    # Shared memory allocated per block (Bytes)
├── Occupancy Analysis
│   ├── sm__maximum_warps_per_active_cycle_pct    # Theoretical occupancy ceiling (%)
│   └── sm__warps_active.avg.pct_of_peak_sustained_active # Achieved occupancy during execution (%)
├── Throughput & Compute
│   ├── sm__ops_path_tensor_src_fp32.sum          # Dispatched FP32 arithmetic instructions
│   └── sm__throughput.avg.pct_of_peak_sustained_elapsed # Peak SM utilization (%)
├── Memory Subsystem
│   ├── dram__bytes.sum.per_second                # Sustained DRAM bandwidth (Bytes/sec)
│   └── gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed # Bus saturation vs 192.0 GB/s peak (%)
└── Warp Execution & Stalls
    ├── smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct # DRAM load dependency stalls
    ├── smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct# Shared memory dependency stalls
    ├── smsp__warp_issue_stalled_barrier_per_warp_active.pct         # __syncthreads() barrier stalls
    ├── smsp__warp_issue_stalled_wait_per_warp_active.pct            # Pipeline execution latency
    └── smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct# Saturated ALU pipeline stalls
```

### Key Architectural Diagnoses

1. **Kernel Launch Overhead vs. Workload Granularity**:
   Nsight Systems timeline analysis reveals that `cudaLaunchKernel` introduces an average CPU dispatch latency of **$12.6\ \mu\text{s} - 24.8\ \mu\text{s}$**. For tiny problem sizes ($64 \times 64$ GEMM taking $3.33\ \mu\text{s}$), CPU overhead dominates total wall-clock time by $7.4\times$. For production deployments, small kernels must be batched or captured into **CUDA Graphs** (`cudaGraph_t`).

2. **The Occupancy Fallacy (High Occupancy $\neq$ High Performance)**:
   - Baseline Naive GEMM achieves near **$100\%$ theoretical occupancy** due to zero shared-memory allocation and low register counts. However, it achieves only **$396\text{ GFLOP/s}$** because warps spend $>70\%$ of active cycles stalled on `long_scoreboard` waiting for DRAM.
   - Tiled GEMM allocates $2\text{ KB}$ of shared memory per block, lowering theoretical occupancy, but sustains **$560\text{ GFLOP/s}$ ($+41\%$ faster)** because cached data in SRAM prevents memory stall cycles.
   - **Conclusion**: Increasing Instruction-Level Parallelism (ILP) and caching data in SRAM hides latency more effectively than maintaining large numbers of idle resident warps.

3. **SM Wave Scheduling on RTX 3050 (16 SMs)**:
   The GA107 GPU features 16 SMs. Thread block grids should always be sized as integer multiples of 16 blocks to prevent "tail-wave" imbalance, where a final partial wave executes on a fraction of the SMs while the remainder sit idle.

---

## Authors & Acknowledgments

- **Developed by**: Shreejith
- **Repository**: [gpu-roofline-analysis](https://github.com/Shreejith-19/gpu-roofline-analysis)
- **References**:
  - Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An insightful visual performance model for multicore architectures.* Communications of the ACM.
  - NVIDIA Corporation. *CUDA C++ Programming Guide* & *NVIDIA Nsight Compute CLI Documentation*.