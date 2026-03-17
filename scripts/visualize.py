"""Visualize forgetting curves from comparison experiment results."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Korean font support
KOREAN_FONTS = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
for fp in KOREAN_FONTS:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_recall_curves(results: dict, output_path: str):
    """Recall rate over time for exponential vs power law."""
    exp = results["exponential"]["snapshots"]
    pl = results["power_law"]["snapshots"]

    exp_ticks = [s["tick"] for s in exp]
    exp_recall = [s["recall_rate"] for s in exp]
    pl_ticks = [s["tick"] for s in pl]
    pl_recall = [s["recall_rate"] for s in pl]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_ticks, exp_recall, "o-", color="#2196F3", linewidth=2, markersize=6, label=f"Exponential (score: {results['exponential']['composite_score']:.4f})")
    ax.plot(pl_ticks, pl_recall, "s-", color="#FF5722", linewidth=2, markersize=6, label=f"Power Law (score: {results['power_law']['composite_score']:.4f})")

    ax.set_xlabel("Time (ticks)", fontsize=12)
    ax.set_ylabel("Recall Rate", fontsize=12)
    ax.set_title("Forgetting Curves: Exponential vs Power Law Decay\n(416 memories, ko-sroberta-multitask)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.02, 0.50)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_precision_curves(results: dict, output_path: str):
    """Precision rate over time."""
    exp = results["exponential"]["snapshots"]
    pl = results["power_law"]["snapshots"]

    exp_ticks = [s["tick"] for s in exp]
    exp_prec = [s["precision_rate"] for s in exp]
    pl_ticks = [s["tick"] for s in pl]
    pl_prec = [s["precision_rate"] for s in pl]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_ticks, exp_prec, "o-", color="#2196F3", linewidth=2, markersize=6, label="Exponential")
    ax.plot(pl_ticks, pl_prec, "s-", color="#FF5722", linewidth=2, markersize=6, label="Power Law")

    ax.set_xlabel("Time (ticks)", fontsize=12)
    ax.set_ylabel("Precision Rate", fontsize=12)
    ax.set_title("Precision Over Time: Exponential vs Power Law Decay", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_combined(results: dict, output_path: str):
    """Side-by-side recall + precision."""
    exp = results["exponential"]["snapshots"]
    pl = results["power_law"]["snapshots"]

    ticks_exp = [s["tick"] for s in exp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Recall
    ax1.plot(ticks_exp, [s["recall_rate"] for s in exp], "o-", color="#2196F3", lw=2, ms=5, label="Exponential")
    ax1.plot([s["tick"] for s in pl], [s["recall_rate"] for s in pl], "s-", color="#FF5722", lw=2, ms=5, label="Power Law")
    ax1.set_title("Recall Rate", fontsize=13)
    ax1.set_xlabel("Ticks")
    ax1.set_ylabel("Recall")
    ax1.set_ylim(-0.02, 0.50)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision
    ax2.plot(ticks_exp, [s["precision_rate"] for s in exp], "o-", color="#2196F3", lw=2, ms=5, label="Exponential")
    ax2.plot([s["tick"] for s in pl], [s["precision_rate"] for s in pl], "s-", color="#FF5722", lw=2, ms=5, label="Power Law")
    ax2.set_title("Precision Rate", fontsize=13)
    ax2.set_xlabel("Ticks")
    ax2.set_ylabel("Precision")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Memory Decay Comparison (416 memories, Korean embedding)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_decay_shape_comparison(results: dict, output_path: str):
    """Theoretical decay curves overlaid with observed data."""
    ticks = np.linspace(0, 200, 200)
    exp_theory = np.exp(-0.02 * ticks)
    pl_theory = 1.0 / (1.0 + 0.08 * ticks) ** 1.5

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ticks, exp_theory, "-", color="#2196F3", linewidth=2, label="Exponential: $e^{-\\lambda t}$")
    ax.plot(ticks, pl_theory, "-", color="#FF5722", linewidth=2, label="Power Law: $(1 + \\beta t)^{-\\alpha}$")

    # Overlay actual recall
    exp = results["exponential"]["snapshots"]
    pl = results["power_law"]["snapshots"]
    ax.scatter([s["tick"] for s in exp], [s["recall_rate"] for s in exp],
               color="#2196F3", marker="o", s=50, zorder=5, alpha=0.7, label="Exponential (observed)")
    ax.scatter([s["tick"] for s in pl], [s["recall_rate"] for s in pl],
               color="#FF5722", marker="s", s=50, zorder=5, alpha=0.7, label="Power Law (observed)")

    ax.set_xlabel("Time (ticks)", fontsize=12)
    ax.set_ylabel("Normalized Activation / Recall", fontsize=12)
    ax.set_title("Theoretical Decay Functions vs Observed Recall", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "data/comparison_results_korean_emb.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "docs/figures"

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = load_results(results_path)

    plot_recall_curves(results, f"{out_dir}/recall_curves.png")
    plot_precision_curves(results, f"{out_dir}/precision_curves.png")
    plot_combined(results, f"{out_dir}/combined_comparison.png")
    plot_decay_shape_comparison(results, f"{out_dir}/decay_shape_comparison.png")
