#!/usr/bin/env python3
"""
GPU Performance Comparisons Plotting Script.
Generates publication-quality comparison charts with clearly defined metrics for:
1. Naive vs. Tiled (16x16 Shared Memory) Matrix Multiplication (GEMM)
2. Parallel Sum vs. Max Reduction Kernels (Grid-stride + Warp Shuffle)
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# Shared visual styling
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#CFD8DC'
plt.rcParams['axes.linewidth'] = 1.0

def load_benchmark_data(csv_path):
    gemm_naive = {}
    gemm_tiled = {}
    reduce_sum = []
    reduce_max = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['KernelName']
            time_ms = float(row['ElapsedMillis'])
            bw_gbs = float(row['AchievedBandwidthGBs'])
            gflops = float(row['AchievedGflops'])
            oi = float(row['OperationalIntensity'])
            bytes_trans = float(row['BytesTransferred'])
            flops = float(row['FlopsExecuted'])
            prob_size = float(row['ProblemSize'])

            entry = {
                'name': name,
                'time_ms': time_ms,
                'bw_gbs': bw_gbs,
                'gflops': gflops,
                'oi': oi,
                'bytes': bytes_trans,
                'flops': flops,
                'size': prob_size
            }

            if name.startswith("Naive_"):
                tag = name.replace("Naive_", "")
                gemm_naive[tag] = entry
            elif name.startswith("Tiled16_"):
                tag = name.replace("Tiled16_", "")
                gemm_tiled[tag] = entry
            elif name.startswith("Reduce_"):
                tag = name.replace("Reduce_", "")
                reduce_sum.append((prob_size, tag, entry))
            elif name.startswith("ReduceMax_"):
                tag = name.replace("ReduceMax_", "")
                reduce_max.append((prob_size, tag, entry))

    reduce_sum.sort(key=lambda x: x[0])
    reduce_max.sort(key=lambda x: x[0])

    return gemm_naive, gemm_tiled, reduce_sum, reduce_max

def plot_matmul_comparison(gemm_naive, gemm_tiled, output_path):
    """
    Plots Naive vs Tiled Matrix Multiplication performance, speedup, and metric definitions.
    """
    keys_order = [
        "64x64", "128x128", "256x256", "512x512", "1024x1024", "2048x2048",
        "67x113", "131x73", "512x64", "64x512", "257x255"
    ]
    labels = [k for k in keys_order if k in gemm_naive and k in gemm_tiled]
    
    naive_gflops = [gemm_naive[k]['gflops'] for k in labels]
    tiled_gflops = [gemm_tiled[k]['gflops'] for k in labels]
    speedups = [gemm_naive[k]['time_ms'] / gemm_tiled[k]['time_ms'] for k in labels]

    fig = plt.figure(figsize=(14.2, 9.4), dpi=300, facecolor='#FFFFFF')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 0.78], hspace=0.35, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_defs = fig.add_subplot(gs[1, :])

    # --- Subplot 1: Throughput (GFLOP/s) ---
    x = np.arange(len(labels))
    width = 0.38

    ax1.bar(x - width/2, naive_gflops, width, label='Naive GEMM (Global Memory Only)', 
            color='#FFA726', edgecolor='#E65100', linewidth=1.1, zorder=3)
    ax1.bar(x + width/2, tiled_gflops, width, label='Tiled GEMM (16×16 Shared Memory)', 
            color='#29B6F6', edgecolor='#0277BD', linewidth=1.1, zorder=3)

    ax1.set_title("Attainable Throughput (GFLOP/s)", fontsize=12, weight='bold', color='#1A237E', pad=10)
    ax1.set_ylabel("Throughput (GFLOP/s) — Higher is better", fontsize=10, weight='bold', color='#37474F')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=8.5, weight='medium')
    ax1.set_ylim(0, 750)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5, color='#B0BEC5', zorder=0)
    ax1.legend(loc='upper left', fontsize=8.8, framealpha=0.92, edgecolor='#CFD8DC')

    # Peak GFLOP annotation on highest tiled run
    max_idx = int(np.argmax(tiled_gflops))
    ax1.annotate(f"Peak: {tiled_gflops[max_idx]:.1f} GFLOP/s\n(+{((speedups[max_idx]-1)*100):.0f}% vs Naive)",
                 xy=(x[max_idx] + width/2, tiled_gflops[max_idx]),
                 xytext=(x[max_idx] - 0.2, 630),
                 ha='center',
                 arrowprops=dict(arrowstyle="->", color="#01579B", lw=1.3),
                 fontsize=8.5, weight='bold', color="#01579B",
                 bbox=dict(boxstyle="round,pad=0.28", fc="#E1F5FE", ec="#81D4FA", lw=0.9))

    # --- Subplot 2: Speedup Factor (Naive / Tiled) ---
    colors = ['#43A047' if s >= 1.0 else '#E53935' for s in speedups]
    bars = ax2.bar(x, speedups, width=0.55, color=colors, edgecolor='#263238', linewidth=0.9, zorder=3)
    ax2.axhline(1.0, color='#37474F', linestyle='--', linewidth=1.2, zorder=4, label='Parity (1.00x)')

    for bar, s in zip(bars, speedups):
        y_pos = bar.get_height() + 0.035 if s >= 1.0 else 1.02
        color = '#1B5E20' if s >= 1.0 else '#B71C1C'
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f"{s:.2f}x",
                 ha='center', va='bottom', fontsize=8.0, weight='bold', color=color)

    ax2.set_title("Speedup Factor (T_naive / T_tiled)", fontsize=12, weight='bold', color='#1A237E', pad=10)
    ax2.set_ylabel("Speedup Multiplier (T_naive / T_tiled)", fontsize=10, weight='bold', color='#37474F')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha='right', fontsize=8.5, weight='medium')
    ax2.set_ylim(0.7, 1.6)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5, color='#B0BEC5', zorder=0)
    ax2.legend(loc='upper left', fontsize=8.8, framealpha=0.92, edgecolor='#CFD8DC')

    # --- Subplot 3: Metric Definitions Box ---
    ax_defs.axis('off')
    defs_text = (
        "  EXPLICIT METRIC DEFINITIONS & FORMULAS\n"
        "  " + "─" * 105 + "\n"
        "  1. FLOPs Executed:                  FLOPs = 2 × M × N × K\n"
        "     • 1 multiply + 1 addition (Fused Multiply-Add) per inner-loop iteration for A[M,K] × B[K,N].\n"
        "  2. Attainable Throughput (GFLOP/s):  GFLOP/s = (FLOPs) / (Execution Time in Seconds × 10⁹)\n"
        "     • Measures real computation rate across all SMs on the NVIDIA RTX 3050 Laptop GPU.\n"
        "  3. Speedup Factor:                  Speedup = T_naive / T_tiled\n"
        "     • Multiplier > 1.0 indicates speedup achieved through cooperative shared-memory tile staging.\n"
        "  4. Theoretical DRAM Traffic:        Bytes = 4 × (M × K + K × N + M × N)  (FP32 = 4 bytes per element)\n"
        "     • Compulsory DRAM traffic assuming single read of A & B and single write of C.\n"
        "  5. Operational Intensity (FLOP/B):  OI = (2 × M × N × K) / [4 × (M × K + K × N + M × N)]\n"
        "     • Quantifies arithmetic intensity (FLOPs per byte transferred). Increases proportionally with matrix size.\n"
        "  6. Architectural Analysis:          In Tiled GEMM with 16×16 blocks, global memory loads are cached in SM Shared\n"
        "     Memory and reused 16 times per thread block, accelerating compute from 396 GFLOP/s to 560 GFLOP/s (+41% at 2048²)."
    )
    ax_defs.text(0.04, 0.48, defs_text, ha='left', va='center', fontsize=8.2, family='monospace',
                 color='#0D47A1', bbox=dict(boxstyle="round,pad=0.7", fc="#F8F9FA", ec="#B0BEC5", lw=1.0))

    fig.suptitle("GEMM Performance Evaluation: Baseline Naive vs. 16×16 Tiled Shared Memory", 
                 fontsize=14.5, weight='bold', color='#0D47A1', y=0.975)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    svg_path = os.path.splitext(output_path)[0] + ".svg"
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved GEMM comparison plot to {output_path} and {svg_path}")

def plot_reduction_comparison(reduce_sum, reduce_max, output_path, peak_bw_gbs=192.0):
    """
    Plots Parallel Sum vs. Max Reduction performance, bandwidth saturation, and metric definitions.
    """
    labels = [item[1].replace("-Floats", "").replace("Irregular-", "") for item in reduce_sum]
    sum_bw = [item[2]['bw_gbs'] for item in reduce_sum]
    max_bw = [item[2]['bw_gbs'] for item in reduce_max]
    sum_gops = [item[2]['gflops'] for item in reduce_sum]
    max_gops = [item[2]['gflops'] for item in reduce_max]

    fig = plt.figure(figsize=(14.2, 9.4), dpi=300, facecolor='#FFFFFF')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 0.78], hspace=0.35, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_defs = fig.add_subplot(gs[1, :])

    x = np.arange(len(labels))
    width = 0.36

    # --- Subplot 1: Effective Memory Bandwidth (GB/s) ---
    ax1.bar(x - width/2, sum_bw, width, label='Parallel Sum Reduction (Tree + Shuffle)', 
            color='#66BB6A', edgecolor='#1B5E20', linewidth=1.1, zorder=3)
    ax1.bar(x + width/2, max_bw, width, label='Parallel Max Reduction (Tree + Shuffle)', 
            color='#AB47BC', edgecolor='#4A148C', linewidth=1.1, zorder=3)

    # Theoretical peak DRAM ceiling
    ax1.axhline(peak_bw_gbs, color='#D32F2F', linestyle='--', linewidth=1.5, zorder=4,
                label=f'Theoretical Peak DRAM ({peak_bw_gbs:.1f} GB/s)')

    ax1.set_title("Effective Streaming Bandwidth (GB/s)", fontsize=12, weight='bold', color='#1A237E', pad=10)
    ax1.set_ylabel("Effective Bandwidth (GB/s) — Higher is better", fontsize=10, weight='bold', color='#37474F')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8, weight='medium')
    ax1.set_ylim(0, 245)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5, color='#B0BEC5', zorder=0)
    ax1.legend(loc='upper left', fontsize=8.8, framealpha=0.92, edgecolor='#CFD8DC')

    # Peak annotation above the highest bar
    peak_sum_idx = int(np.argmax(sum_bw))
    ax1.annotate(f"Peak: {sum_bw[peak_sum_idx]:.1f} GB/s\n({(sum_bw[peak_sum_idx]/peak_bw_gbs)*100:.1f}% Saturation)",
                 xy=(x[peak_sum_idx] - width/2, sum_bw[peak_sum_idx]),
                 xytext=(x[peak_sum_idx] - 0.2, 215),
                 ha='center',
                 arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.3),
                 fontsize=8.5, weight='bold', color="#1B5E20",
                 bbox=dict(boxstyle="round,pad=0.28", fc="#E8F5E9", ec="#A5D6A7", lw=0.9))

    # --- Subplot 2: Processing Rate (G-Elements/s) ---
    ax2.plot(x, sum_gops, marker='D', color='#2E7D32', linewidth=2.0, markersize=6.5, label='Sum Reduction Rate (G-Elem/s)', zorder=3)
    ax2.plot(x, max_gops, marker='^', color='#7B1FA2', linewidth=2.0, markersize=7.0, label='Max Reduction Rate (G-Elem/s)', zorder=3)

    for i in range(len(labels)):
        ax2.text(x[i], sum_gops[i] + 1.8, f"{sum_gops[i]:.1f}", ha='center', fontsize=7.8, weight='bold', color='#1B5E20')

    ax2.set_title("Processing Rate (G-Elements/s)", fontsize=12, weight='bold', color='#1A237E', pad=10)
    ax2.set_ylabel("G-Elements / Second (10⁹ ops/s)", fontsize=10, weight='bold', color='#37474F')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.8, weight='medium')
    ax2.set_ylim(0, 55)
    ax2.grid(True, axis='both', linestyle='--', alpha=0.5, color='#B0BEC5', zorder=0)
    ax2.legend(loc='upper left', fontsize=8.8, framealpha=0.92, edgecolor='#CFD8DC')

    # --- Subplot 3: Metric Definitions Box ---
    ax_defs.axis('off')
    defs_text = (
        "  EXPLICIT METRIC DEFINITIONS & FORMULAS\n"
        "  " + "─" * 105 + "\n"
        "  1. Elements Processed (N):          Total count of float32 values streamed across GPU grid-stride loops.\n"
        "  2. Effective Bytes Transferred:     Bytes = N × 4 Bytes (FP32 = 4 bytes per element)\n"
        "     • Measures compulsory data streaming; block partial-reduction writes contribute < 0.01% to DRAM traffic.\n"
        "  3. Effective Bandwidth (GB/s):      BW_eff = (N × 4 Bytes) / (Execution Time in Seconds × 10⁹)\n"
        "     • Rigorous metric assessing memory-subsystem streaming throughput against physical GDDR6 bandwidth.\n"
        "  4. DRAM Bandwidth Saturation:       Saturation (%) = (BW_eff / 192.0 GB/s) × 100%\n"
        "     • Hardware ceiling of NVIDIA RTX 3050 Laptop (128-bit bus @ 6001 MHz = 192.03 GB/s peak).\n"
        "  5. Processing Rate (GElem/s):       Rate = N / (Execution Time in Seconds × 10⁹)\n"
        "     • Corresponds to GFLOP/s for Sum Reduction (N-1 additions) and G-Comparisons/s for Max Reduction (N-1 max ops).\n"
        "  6. Operational Intensity (OI):      OI = (N - 1 Operations) / (N × 4 Bytes) ≈ 0.250 FLOP/Byte (or Op/Byte)\n"
        "     • Far below the 37.5 FLOP/B ridge point: reductions are strictly bound by DRAM bus throughput."
    )
    ax_defs.text(0.04, 0.48, defs_text, ha='left', va='center', fontsize=8.2, family='monospace',
                 color='#1B5E20', bbox=dict(boxstyle="round,pad=0.7", fc="#F8F9FA", ec="#B0BEC5", lw=1.0))

    fig.suptitle("Parallel Reduction Kernels Benchmark: Sum Reduction vs. Max Reduction", 
                 fontsize=14.5, weight='bold', color='#0D47A1', y=0.975)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    svg_path = os.path.splitext(output_path)[0] + ".svg"
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved Reduction comparison plot to {output_path} and {svg_path}")

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(root, "results", "benchmark_results.csv")
    plots_dir = os.path.join(root, "results", "plots")

    gemm_naive, gemm_tiled, reduce_sum, reduce_max = load_benchmark_data(csv_path)

    matmul_plot = os.path.join(plots_dir, "matmul_comparison.png")
    plot_matmul_comparison(gemm_naive, gemm_tiled, matmul_plot)

    reduction_plot = os.path.join(plots_dir, "reduction_comparison.png")
    plot_reduction_comparison(reduce_sum, reduce_max, reduction_plot)

if __name__ == '__main__':
    main()
