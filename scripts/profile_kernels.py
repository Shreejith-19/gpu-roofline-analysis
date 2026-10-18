#!/usr/bin/env python3
"""
GPU Kernel Profiling Workflow Script using NVIDIA Nsight Systems and Nsight Compute.
Automates timeline tracing, microarchitectural analysis, and Roofline profiling for:
  1. matmulNaiveKernel (Baseline Naive Matrix Multiplication)
  2. matmulTiledKernel (16x16 Shared-Memory Tiled GEMM)
  3. reduceBlockKernel (Parallel Sum Reduction with Warp Shuffle)
  4. reduceMaxBlockKernel (Parallel Max Reduction with Warp Shuffle)

Tailored for NVIDIA GeForce RTX 3050 Laptop GPU (Ampere sm_86, 16 SMs, 4 GB VRAM).
"""

import os
import sys
import subprocess
import argparse
import shutil

# Executable mapping
BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "build", "tests", "Release")
BIN_BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "build", "Release")

KERNEL_CONFIGS = {
    "matmul_naive": {
        "binary": os.path.join(BIN_DIR, "test_matmul_naive.exe"),
        "kernel_name": "matmulNaiveKernel",
        "desc": "Baseline Naive GEMM (Global Memory Only)"
    },
    "matmul_tiled": {
        "binary": os.path.join(BIN_DIR, "test_matmul_tiled.exe"),
        "kernel_name": "regex:matmulTiledKernel.*",
        "desc": "Tiled GEMM (16x16 Shared Memory)"
    },
    "reduce_sum": {
        "binary": os.path.join(BIN_DIR, "test_reduce_sum.exe"),
        "kernel_name": "regex:reduceBlockKernel.*",
        "desc": "Parallel Float32 Sum Reduction"
    },
    "reduce_max": {
        "binary": os.path.join(BIN_DIR, "test_reduce_max.exe"),
        "kernel_name": "regex:reduceMaxBlockKernel.*",
        "desc": "Parallel Float32 Max Reduction"
    }
}

# Verified Nsight Compute 2026.3.1.0 metrics for Ampere GA107 (sm_86)
NCU_CORE_METRICS = [
    # Resource Allocation
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_allocated",
    # Occupancy
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # Compute Throughput
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_src_fp32.sum",
    # DRAM Bandwidth
    "dram__bytes.sum",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    # Dominant Warp Stalls
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
]

def check_profiler_availability():
    nsys_path = shutil.which("nsys")
    ncu_path = shutil.which("ncu")
    
    print("=" * 80)
    print("            NVIDIA PROFILER ENVIRONMENT VERIFICATION")
    print("=" * 80)
    
    if nsys_path:
        res = subprocess.run(["nsys", "--version"], stdout=subprocess.PIPE, text=True)
        print(f"[FOUND] Nsight Systems: {res.stdout.strip()}")
    else:
        print("[MISSING] Nsight Systems (nsys) not found on PATH.")

    if ncu_path:
        res = subprocess.run(["ncu", "--version"], stdout=subprocess.PIPE, text=True)
        first_line = res.stdout.strip().split("\n")[0]
        print(f"[FOUND] Nsight Compute: {first_line}")
    else:
        print("[MISSING] Nsight Compute (ncu) not found on PATH.")
        
    print(" Target Device     : NVIDIA GeForce RTX 3050 Laptop GPU (Ampere sm_86)")
    print(" Compute SMs       : 16 Streaming Multiprocessors (128 FP32 cores/SM = 2048 FP32 cores)")
    print(" VRAM Capacity     : 4.00 GB GDDR6 (128-bit bus @ 6001 MHz = 192.03 GB/s peak)")
    print(" Peak FP32 Compute : 7.2 TFLOP/s (7200 GFLOP/s)")
    print("=" * 80)
    return bool(nsys_path), bool(ncu_path)

def profile_nsys(kernel_key, out_dir):
    cfg = KERNEL_CONFIGS[kernel_key]
    binary = cfg["binary"]
    if not os.path.exists(binary):
        print(f"[ERROR] Binary not found: {binary}. Please build the project first.")
        return

    os.makedirs(out_dir, exist_ok=True)
    report_base = os.path.join(out_dir, f"nsys_{kernel_key}")
    rep_file = f"{report_base}.nsys-rep"

    print(f"\n>>> Running Nsight Systems Timeline Trace on [{cfg['desc']}]...")
    cmd_trace = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "-o", report_base,
        "--force-overwrite=true",
        binary
    ]
    subprocess.run(cmd_trace)

    if os.path.exists(rep_file):
        print(f"\n--- Kernel Execution Summary for {kernel_key} ---")
        cmd_stats_kern = [
            "nsys", "stats",
            "--force-export=true",
            "--report", "cuda_gpu_kern_sum",
            rep_file
        ]
        subprocess.run(cmd_stats_kern)

        print(f"\n--- CUDA API Summary (Launch Overhead) for {kernel_key} ---")
        cmd_stats_api = [
            "nsys", "stats",
            "--force-export=true",
            "--report", "cuda_api_sum",
            rep_file
        ]
        subprocess.run(cmd_stats_api)

def profile_ncu(kernel_key, out_dir, mode="metrics"):
    cfg = KERNEL_CONFIGS[kernel_key]
    binary = cfg["binary"]
    if not os.path.exists(binary):
        print(f"[ERROR] Binary not found: {binary}. Please build the project first.")
        return

    os.makedirs(out_dir, exist_ok=True)
    report_base = os.path.join(out_dir, f"ncu_{kernel_key}")

    print(f"\n>>> Running Nsight Compute [{mode.upper()}] Profiling on [{cfg['desc']}]...")
    
    if mode == "roofline":
        cmd = [
            "ncu",
            "--kernel-name", cfg["kernel_name"],
            "--launch-count", "1",
            "--section", "SpeedOfLight_RooflineChart",
            "-o", report_base,
            "--force-overwrite",
            binary
        ]
    elif mode == "detailed":
        cmd = [
            "ncu",
            "--kernel-name", cfg["kernel_name"],
            "--launch-count", "1",
            "--set", "detailed",
            "-o", report_base,
            "--force-overwrite",
            binary
        ]
    else: # core verified metrics
        metrics_str = ",".join(NCU_CORE_METRICS)
        cmd = [
            "ncu",
            "--kernel-name", cfg["kernel_name"],
            "--launch-count", "1",
            "--metrics", metrics_str,
            "--page", "raw",
            binary
        ]

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("\n" + "!" * 80)
        print("[NOTE] Nsight Compute returned non-zero. If you encountered ERR_NVGPUCTRPERM:")
        print("  1. Open NVIDIA Control Panel -> Developer Menu -> 'Manage GPU Performance Counters'")
        print("     -> Select 'Allow access to the GPU performance counters to all users'.")
        print("  2. Alternatively, launch your PowerShell terminal as Administrator.")
        print("!" * 80)

def main():
    parser = argparse.ArgumentParser(description="Profile CUDA Kernels with Nsight Systems & Compute")
    parser.add_argument("--kernel", choices=["matmul_naive", "matmul_tiled", "reduce_sum", "reduce_max", "all"], 
                        default="all", help="Kernel workload to profile")
    parser.add_argument("--tool", choices=["nsys", "ncu", "all"], default="nsys", 
                        help="Profiler tool to execute")
    parser.add_argument("--ncu-mode", choices=["metrics", "roofline", "detailed"], default="metrics",
                        help="Mode for Nsight Compute profiling")
    parser.add_argument("--out-dir", default=os.path.join("results", "profiles"), 
                        help="Directory for profile reports")
    args = parser.parse_args()

    has_nsys, has_ncu = check_profiler_availability()

    kernels = list(KERNEL_CONFIGS.keys()) if args.kernel == "all" else [args.kernel]

    for k in kernels:
        if args.tool in ["nsys", "all"] and has_nsys:
            profile_nsys(k, args.out_dir)
        if args.tool in ["ncu", "all"] and has_ncu:
            profile_ncu(k, args.out_dir, mode=args.ncu_mode)

if __name__ == '__main__':
    main()
