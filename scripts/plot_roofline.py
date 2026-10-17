#!/usr/bin/env python3
"""
GPU Roofline Model Plotting Script.
Reads benchmark CSV results and generates publication-grade empirical Roofline curves
comparing measured operational intensity vs. theoretical hardware ceilings.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_roofline(csv_file, output_path, peak_gflops=7200.0, peak_bw_gbs=192.0):
    """
    Plots an empirical Roofline curve.
    
    :param csv_file: Path to input CSV benchmark results
    :param output_path: Destination path for saved PNG/SVG plot
    :param peak_gflops: Theoretical peak FP32 GFLOP/s (default RTX 3050 mobile ~7.2 TFLOP/s)
    :param peak_bw_gbs: Theoretical peak memory bandwidth GB/s (default RTX 3050 GDDR6 ~192 GB/s)
    """
    # 1. Operational Intensity domain (FLOPs / Byte)
    oi_range = np.logspace(-2, 3, num=500)
    
    # 2. Roofline ceilings: Min(Peak GFLOP/s, Peak Bandwidth * OI)
    mem_bound = peak_bw_gbs * oi_range
    roofline_ceiling = np.minimum(peak_gflops, mem_bound)
    ridge_point = peak_gflops / peak_bw_gbs

    # 3. Setup styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot theoretical ceilings
    ax.loglog(oi_range, roofline_ceiling, 'k-', linewidth=2.5, label='Theoretical Ceiling')
    ax.axvline(x=ridge_point, color='gray', linestyle='--', alpha=0.7, 
               label=f'Ridge Point ({ridge_point:.2f} FLOP/Byte)')

    # Add ceiling boundary annotations
    ax.text(oi_range[10], peak_bw_gbs * oi_range[10] * 1.3, 
            f'Memory Bandwidth Limit ({peak_bw_gbs:.1f} GB/s)', 
            rotation=38, fontsize=9, color='darkblue', weight='bold')
    ax.text(oi_range[-100], peak_gflops * 1.08, 
            f'Compute Limit ({peak_gflops:.0f} GFLOP/s)', 
            fontsize=9, color='darkred', weight='bold')

    # 4. Parse CSV data points if file exists
    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    kernel = row['KernelName']
                    oi = float(row['OperationalIntensity'])
                    perf = float(row['AchievedGflops'])
                    ax.scatter(oi, perf, s=80, zorder=5, label=f"Kernel: {kernel}")
        except Exception as e:
            print(f"[WARN] Failed to parse {csv_file}: {e}")

    ax.set_title("NVIDIA GPU Empirical Roofline Model", fontsize=14, pad=15, weight='bold')
    ax.set_xlabel("Operational Intensity (FLOPs / Byte)", fontsize=11, labelpad=10)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=11, labelpad=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Roofline plot saved to: {output_path}")

if __name__ == '__main__':
    default_csv = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark_results.csv")
    default_out = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "roofline_model.png")
    plot_roofline(default_csv, default_out)
