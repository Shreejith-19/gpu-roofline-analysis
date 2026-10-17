#!/usr/bin/env python3
"""
Benchmark Orchestration Script.
Builds the CUDA project using CMake and executes benchmarks,
optionally collecting hardware performance counters via NVIDIA Nsight Compute (ncu).
"""

import os
import subprocess
import sys
import argparse

def run_command(cmd, cwd=None):
    print(f"[EXEC] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd, shell=True)
    if res.returncode != 0:
        print(f"[FAIL] Command exited with code {res.returncode}")
        sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser(description="Orchestrate GPU Roofline Benchmarks")
    parser.add_argument("--build", action="store_true", help="Configure and build project before running")
    parser.add_argument("--profile", action="store_true", help="Profile using Nsight Compute (ncu)")
    parser.add_argument("--config", default="Release", help="Build configuration (Debug/Release)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_dir = os.path.join(project_root, "build")

    if args.build:
        print("[INFO] Configuring and building project...")
        os.makedirs(build_dir, exist_ok=True)
        run_command(f'cmake -B "{build_dir}" -S "{project_root}" -DCMAKE_BUILD_TYPE={args.config}')
        run_command(f'cmake --build "{build_dir}" --config {args.config}')

    smoke_bin = os.path.join(build_dir, "tests", args.config, "smoke_test.exe")
    if not os.path.exists(smoke_bin):
        # Check direct path if single-configuration generator was used
        smoke_bin = os.path.join(build_dir, "tests", "smoke_test.exe")

    if os.path.exists(smoke_bin):
        print(f"[INFO] Executing smoke test binary: {smoke_bin}")
        if args.profile:
            run_command(f'ncu --metrics dram__bytes.sum,sm__ops_path_tensor_src_fp32.sum "{smoke_bin}"')
        else:
            run_command(f'"{smoke_bin}"')
    else:
        print("[WARN] Smoke test binary not found. Run with --build first.")

if __name__ == "__main__":
    main()
