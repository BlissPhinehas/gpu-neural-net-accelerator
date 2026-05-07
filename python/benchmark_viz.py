import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── Parse the raw benchmark_results.txt file ──────────────────────────────────

def parse_results(filepath):
    matmul = {"sizes": [], "cpu": [], "gpu": [], "speedup": []}
    relu   = {"sizes": [], "cpu": [], "gpu": [], "speedup": []}
    softmax= {"sizes": [], "cpu": [], "gpu": [], "speedup": []}

    current = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if "Matrix Multiplication" in line:
                current = matmul
            elif "ReLU" in line:
                current = relu
            elif "Softmax" in line:
                current = softmax
            elif current is not None:
                parts = line.split()
                # Valid data lines have 4 numeric-ish parts
                if len(parts) >= 4:
                    try:
                        size    = int(parts[0])
                        cpu_ms  = float(parts[1])
                        gpu_ms  = float(parts[2])
                        speedup = float(parts[3])
                        current["sizes"].append(size)
                        current["cpu"].append(cpu_ms)
                        current["gpu"].append(gpu_ms)
                        current["speedup"].append(speedup)
                    except ValueError:
                        pass  # skip header/separator lines

    return matmul, relu, softmax


# ── Plot helpers ───────────────────────────────────────────────────────────────

# Clean style — no gridlines clutter, readable fonts
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

CPU_COLOR     = "#4C72B0"  # blue
GPU_COLOR     = "#DD8452"  # orange
SPEEDUP_COLOR = "#55A868"  # green


def plot_operation(ax_time, ax_speedup, data, xlabel, title):
    """
    Left axis:  CPU vs GPU time (ms) as grouped bars
    Right axis: speedup as a line with markers
    """
    x      = np.arange(len(data["sizes"]))
    width  = 0.35

    # Grouped bars — CPU and GPU side by side
    bars_cpu = ax_time.bar(x - width/2, data["cpu"], width,
                            label="CPU", color=CPU_COLOR, alpha=0.85)
    bars_gpu = ax_time.bar(x + width/2, data["gpu"], width,
                            label="GPU", color=GPU_COLOR, alpha=0.85)

    ax_time.set_yscale("log")  # log scale because CPU times dwarf GPU times
    ax_time.set_ylabel("Time (ms, log scale)")
    ax_time.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([str(s) for s in data["sizes"]], rotation=30)
    ax_time.set_xlabel(xlabel)
    ax_time.legend(loc="upper left", framealpha=0.7)

    # Speedup line on the right y-axis
    # Speedup line on the right y-axis
    ax_sp = ax_speedup
    ax_sp.plot(x, data["speedup"], color=SPEEDUP_COLOR,
               marker="o", linewidth=2, markersize=6, label="Speedup")
    ax_sp.set_ylabel("Speedup (×)", color=SPEEDUP_COLOR)
    ax_sp.tick_params(axis="y", labelcolor=SPEEDUP_COLOR)
    ax_sp.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f×"))
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels([str(s) for s in data["sizes"]], rotation=30)
    ax_sp.set_xlabel(xlabel)

    # Annotate each speedup point
    for i, (xi, sp) in enumerate(zip(x, data["speedup"])):
        ax_sp.annotate(f"{sp:.0f}×",
                       xy=(xi, sp),
                       xytext=(0, 8),
                       textcoords="offset points",
                       ha="center",
                       fontsize=9,
                       color=SPEEDUP_COLOR)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    results_path = os.path.join(os.path.dirname(__file__),
                                "../plots/benchmark_results.txt")
    output_path  = os.path.join(os.path.dirname(__file__),
                                "../plots/")

    matmul, relu, softmax = parse_results(results_path)

    # One figure with 3 rows — one per operation
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    fig.suptitle("GPU vs CPU Performance — Tesla T4",
                 fontsize=15, fontweight="bold", y=0.98)

    plot_operation(axes[0][0], axes[0][1], matmul,
                   "Matrix size (N×N)", "Matrix multiplication")

    plot_operation(axes[1][0], axes[1][1], relu,
                   "Number of elements", "ReLU activation")

    plot_operation(axes[2][0], axes[2][1], softmax,
                   "Number of elements", "Softmax")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save as high-res PNG
    out = os.path.join(output_path, "performance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()