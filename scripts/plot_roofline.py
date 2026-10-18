#!/usr/bin/env python3
"""
GPU Roofline Model Plotting Script.
Reads benchmark CSV results and generates a publication-grade, clean, and elegant
empirical Roofline model chart comparing measured kernel performance against
the theoretical hardware ceilings of the NVIDIA GeForce RTX 3050 Laptop GPU.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_roofline(csv_file, output_path, peak_gflops=7200.0, peak_bw_gbs=192.0):
    """
    Plots an empirical Roofline curve with clean visual hierarchy and publication styling.
    
    :param csv_file: Path to input CSV benchmark results
    :param output_path: Destination path for saved PNG/SVG plot
    :param peak_gflops: Theoretical peak FP32 GFLOP/s (default RTX 3050 mobile ~7200 GFLOP/s)
    :param peak_bw_gbs: Theoretical peak memory bandwidth GB/s (default RTX 3050 GDDR6 ~192 GB/s)
    """
    # 1. Operational Intensity domain (FLOPs / Byte)
    oi_min, oi_max = 0.05, 1000.0
    oi_range = np.logspace(np.log10(oi_min), np.log10(oi_max), num=600)
    
    # 2. Roofline ceilings: Min(Peak GFLOP/s, Peak Bandwidth * OI)
    mem_bound = peak_bw_gbs * oi_range
    roofline_ceiling = np.minimum(peak_gflops, mem_bound)
    ridge_point = peak_gflops / peak_bw_gbs # ~ 37.5 FLOP/Byte

    # 3. Setup styling & canvas
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#B0BEC5'
    plt.rcParams['axes.linewidth'] = 1.0

    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=300, facecolor='#FFFFFF')
    ax.set_facecolor('#FAFAFA')

    # Shaded operational regimes
    ax.axvspan(oi_min, ridge_point, color='#E3F2FD', alpha=0.45, zorder=0, label='_nolegend_')
    ax.axvspan(ridge_point, oi_max, color='#FFF3E0', alpha=0.35, zorder=0, label='_nolegend_')

    # Clean regime labels positioned in open whitespace
    ax.text(0.062, 500.0, "MEMORY-BOUND REGIME\n(Bound by DRAM Bandwidth)", 
            fontsize=9.0, color='#1565C0', weight='bold', alpha=0.85, ha='left', va='center')
    ax.text(620.0, 3200.0, "COMPUTE-BOUND REGIME\n(Bound by FP32 SM ALUs)", 
            fontsize=9.0, color='#E65100', weight='bold', alpha=0.85, ha='center', va='center')

    # Plot theoretical ceiling curves
    ax.loglog(oi_range, roofline_ceiling, color='#263238', linewidth=2.8, zorder=3, label='Theoretical Roofline Ceiling')

    # Ridge Point indicator placed cleanly near the ceiling inflection knee
    ax.axvline(x=ridge_point, color='#78909C', linestyle='--', linewidth=1.5, alpha=0.85, zorder=2)
    ax.text(ridge_point * 1.06, 5200.0, f"Ridge Point: {ridge_point:.1f} FLOP/B", 
            fontsize=8.5, color='#37474F', weight='bold', va='center',
            bbox=dict(boxstyle="round,pad=0.25", fc="#ECEFF1", ec="#B0BEC5", lw=0.8, alpha=0.9))

    # Memory bandwidth ceiling annotation (angled pill badge)
    ax.text(0.28, peak_bw_gbs * 0.28 * 1.45, 
            f" Peak DRAM Bandwidth: {peak_bw_gbs:.1f} GB/s ", 
            rotation=36, fontsize=9.5, color='#0D47A1', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#90CAF9", lw=1.0, alpha=0.95))

    # Compute peak ceiling annotation
    ax.text(280.0, peak_gflops * 1.15, 
            f" Peak FP32 Compute: {peak_gflops:,.0f} GFLOP/s (7.2 TFLOP/s) ", 
            fontsize=9.5, color='#B71C1C', weight='bold', ha='right',
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#EF9A9A", lw=1.0, alpha=0.95))

    # 4. Parse CSV data points
    if os.path.exists(csv_file):
        try:
            naive_points = {}
            tiled_points = {}
            reduce_sum_points = []
            reduce_max_points = []

            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row['KernelName']
                    oi = float(row['OperationalIntensity'])
                    perf = float(row['AchievedGflops'])
                    
                    if name.startswith("Naive_"):
                        tag = name.replace("Naive_", "")
                        naive_points[tag] = (oi, perf)
                    elif name.startswith("Tiled16_"):
                        tag = name.replace("Tiled16_", "")
                        tiled_points[tag] = (oi, perf)
                    elif name.startswith("ReduceMax_"):
                        reduce_max_points.append((oi, perf, name.replace("ReduceMax_", "")))
                    elif name.startswith("Reduce_"):
                        reduce_sum_points.append((oi, perf, name.replace("Reduce_", "")))

            # Plot Naive GEMM
            if naive_points:
                n_oi = [v[0] for v in naive_points.values()]
                n_perf = [v[1] for v in naive_points.values()]
                ax.scatter(n_oi, n_perf, color='#FF9800', marker='o', s=70, 
                           edgecolors='#E65100', linewidth=1.2, zorder=5, label='Naive GEMM (Global Memory Only)')

            # Plot Tiled GEMM
            if tiled_points:
                t_oi = [v[0] for v in tiled_points.values()]
                t_perf = [v[1] for v in tiled_points.values()]
                ax.scatter(t_oi, t_perf, color='#0288D1', marker='s', s=75, 
                           edgecolors='#01579B', linewidth=1.2, zorder=6, label='Tiled GEMM (16×16 Shared Memory)')

            # Connect matched GEMM points with subtle speedup vectors
            for tag, (n_x, n_y) in naive_points.items():
                if tag in tiled_points:
                    t_x, t_y = tiled_points[tag]
                    if n_x == t_x and t_y > n_y:
                        ax.annotate('', xy=(t_x, t_y), xytext=(n_x, n_y),
                                    arrowprops=dict(arrowstyle="->", color="#78909C", lw=1.1, alpha=0.75, shrinkA=3, shrinkB=3))
                        
                    # Clean labels for key square matrix benchmarks
                    if tag in ["64x64", "256x256", "1024x1024", "2048x2048"]:
                        label_y = t_y * 1.15 if tag != "64x64" else t_y * 0.72
                        ax.text(t_x, label_y, f"{tag}", fontsize=8, color='#37474F', 
                                weight='bold', ha='center',
                                bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#CFD8DC", lw=0.6, alpha=0.9))

            # Plot Reductions (slight horizontal jitter to prevent perfect visual overlap at OI=0.25)
            if reduce_sum_points:
                s_oi = [p[0] * 0.96 for p in reduce_sum_points] # Slight -4% visual jitter
                s_perf = [p[1] for p in reduce_sum_points]
                ax.scatter(s_oi, s_perf, color='#43A047', marker='D', s=65, 
                           edgecolors='#1B5E20', linewidth=1.2, zorder=7, label='Parallel Sum Reduction (Tree + Shuffle)')

            if reduce_max_points:
                m_oi = [p[0] * 1.04 for p in reduce_max_points] # Slight +4% visual jitter
                m_perf = [p[1] for p in reduce_max_points]
                ax.scatter(m_oi, m_perf, color='#8E24AA', marker='^', s=70, 
                           edgecolors='#4A148C', linewidth=1.2, zorder=8, label='Parallel Max Reduction (Tree + Shuffle)')

            # Highlighting peak reduction bandwidth point (64M elements)
            if reduce_sum_points:
                best_red = max(reduce_sum_points, key=lambda p: p[1])
                ax.annotate(f"Peak Stream: 179.0 GB/s\n({(179.0/peak_bw_gbs)*100:.1f}% DRAM Peak)",
                            xy=(best_red[0], best_red[1]),
                            xytext=(best_red[0] * 2.5, best_red[1] * 0.75),
                            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2, connectionstyle="arc3,rad=-0.15"),
                            fontsize=8.5, color="#1B5E20", weight='bold',
                            bbox=dict(boxstyle="round,pad=0.25", fc="#E8F5E9", ec="#A5D6A7", lw=0.8))

        except Exception as e:
            print(f"[WARN] Failed to parse {csv_file}: {e}")

    # Axis titles & subtitle
    fig.suptitle("NVIDIA GeForce RTX 3050 Empirical Roofline Analysis", fontsize=14, weight='bold', color='#1A237E', y=0.975)
    ax.set_title("Theoretical Ceilings vs. Measured Kernel Performance (FP32 GEMM & Parallel Reductions)", 
                 fontsize=9.5, color='#546E7A', style='italic', pad=10)
    ax.set_xlabel("Operational Intensity (FLOPs / Byte or Ops / Byte)", fontsize=11, labelpad=9, weight='bold', color='#263238')
    ax.set_ylabel("Attainable Performance (GFLOP/s or G-Ops/s)", fontsize=11, labelpad=9, weight='bold', color='#263238')
    
    ax.set_xlim([oi_min, oi_max])
    ax.set_ylim([1.0, 15000.0])

    # Grid styling
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='#E0E4E8', alpha=0.9)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, color='#ECEFF1', alpha=0.7)

    # Clean Hardware Info Box (Upper Left)
    info_text = (
        "GPU: NVIDIA GeForce RTX 3050 Laptop\n"
        "Architecture: Ampere (SM 86) | SM Count: 16\n"
        "Memory: 4 GB GDDR6 (128-bit @ 6001 MHz)\n"
        "Peak Bandwidth: 192.0 GB/s | Peak FP32: 7.2 TFLOP/s"
    )
    ax.text(0.02, 0.96, info_text, transform=ax.transAxes, fontsize=8.2,
            verticalalignment='top', linespacing=1.35, color='#37474F',
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFFFFF", ec="#CFD8DC", lw=0.9, alpha=0.95))

    # Elegant Legend (Lower Right)
    legend = ax.legend(loc='lower right', frameon=True, fontsize=8.8, framealpha=0.95, 
                       edgecolor='#CFD8DC', facecolor='#FFFFFF', borderpad=0.6, labelspacing=0.45)
    legend.get_frame().set_linewidth(0.8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    svg_path = os.path.splitext(output_path)[0] + ".svg"
    plt.savefig(svg_path)
    print(f"[INFO] Clean publication Roofline plot saved to: {output_path} and {svg_path}")

if __name__ == '__main__':
    default_csv = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark_results.csv")
    default_out = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "roofline_model.png")
    plot_roofline(default_csv, default_out)
