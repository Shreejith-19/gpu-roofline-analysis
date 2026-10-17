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
            naive_oi, naive_perf, naive_labels = [], [], []
            tiled_oi, tiled_perf, tiled_labels = [], [], []

            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    kernel = row['KernelName']
                    oi = float(row['OperationalIntensity'])
                    perf = float(row['AchievedGflops'])
                    
                    if "Naive" in kernel:
                        naive_oi.append(oi)
                        naive_perf.append(perf)
                        naive_labels.append(kernel.replace("Naive_", ""))
                    elif "Tiled" in kernel:
                        tiled_oi.append(oi)
                        tiled_perf.append(perf)
                        tiled_labels.append(kernel.replace("Tiled16_", ""))

            if naive_oi:
                ax.scatter(naive_oi, naive_perf, color='#E65100', marker='o', s=90, 
                           edgecolors='black', linewidth=1.2, zorder=5, label="Naive Kernel (Global Only)")
            if tiled_oi:
                ax.scatter(tiled_oi, tiled_perf, color='#0288D1', marker='s', s=90, 
                           edgecolors='black', linewidth=1.2, zorder=6, label="Tiled Kernel (16x16 Shared)")

            # Connect matched points with light arrows to illustrate speedup
            for i in range(min(len(naive_oi), len(tiled_oi))):
                if naive_oi[i] == tiled_oi[i]:
                    ax.annotate('', xy=(tiled_oi[i], tiled_perf[i]), xytext=(naive_oi[i], naive_perf[i]),
                                arrowprops=dict(arrowstyle="->", color="gray", lw=1, alpha=0.6))
                    # Annotate key matrix sizes
                    dim_str = naive_labels[i]
                    if dim_str in ["64x64", "256x256", "1024x1024", "2048x2048"]:
                        ax.text(tiled_oi[i] * 1.05, tiled_perf[i] * 1.05, dim_str,
                                fontsize=8, color="#37474F", weight='semibold')

        except Exception as e:
            print(f"[WARN] Failed to parse {csv_file}: {e}")

    ax.set_title("NVIDIA GeForce RTX 3050 Empirical Roofline: Naive vs Tiled Matmul", 
                 fontsize=13, pad=15, weight='bold')
    ax.set_xlabel("Operational Intensity (FLOPs / Byte)", fontsize=11, labelpad=10)
    ax.set_ylabel("Attainable Performance (GFLOP/s)", fontsize=11, labelpad=10)
    ax.set_xlim([0.05, 1000])
    ax.set_ylim([1, 12000])
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Roofline plot saved to: {output_path}")

if __name__ == '__main__':
    default_csv = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark_results.csv")
    default_out = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "roofline_model.png")
    plot_roofline(default_csv, default_out)
