"""Visualization script for memory-decay experiment analysis.

Generates 6 figures from experiment results and decay model parameters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = BASE_DIR / "experiments"
OUT_DIR = BASE_DIR / "docs" / "figures" / "analysis"
DPI = 150

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": DPI, "savefig.dpi": DPI})


def load_results(exp_id: str) -> dict:
    path = EXP_DIR / exp_id / "results.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Decay model helpers (mirrors decay.py without importing it)
# ---------------------------------------------------------------------------
def _sigmoid_gate(value: float, center: float, width: float) -> float:
    scaled = (value - center) / max(width, 1e-6)
    if scaled >= 0:
        z = math.exp(-scaled)
        return 1.0 / (1.0 + z)
    z = math.exp(scaled)
    return z / (1.0 + z)


def soft_floor_decay_step(
    activation: float,
    impact: float,
    stability: float,
    *,
    lam: float,
    alpha: float = 2.0,
    rho: float = 0.8,
    floor_min: float = 0.05,
    floor_max: float = 0.35,
    floor_power: float = 2.0,
    gate_center: float = 0.4,
    gate_width: float = 0.08,
    consolidation_gain: float = 0.6,
    min_rate_scale: float = 0.1,
) -> float:
    activation = min(max(float(activation), 0.0), 1.0)
    if activation <= 0.0:
        return 0.0
    impact = min(max(float(impact), 0.0), 1.0)
    stability = max(float(stability), 0.0)
    floor_min = min(max(float(floor_min), 0.0), 1.0)
    floor_max = min(max(float(floor_max), floor_min), 1.0)
    floor_power = max(float(floor_power), 1e-6)
    min_rate_scale = min(max(float(min_rate_scale), 0.0), 1.0)

    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    raw_floor = floor_min + (floor_max - floor_min) * (impact**floor_power)
    floor = min(raw_floor, activation)

    gate = _sigmoid_gate(activation, gate_center, gate_width)
    rate_scale = 1.0 - consolidation_gain * impact * gate
    rate_scale = min(max(rate_scale, min_rate_scale), 1.0)
    effective_rate = max(float(lam) * rate_scale / combined, 0.0)

    updated = floor + (activation - floor) * math.exp(-effective_rate)
    return min(max(updated, floor), activation)


# ---------------------------------------------------------------------------
# 1. Decay Curve Comparison
# ---------------------------------------------------------------------------
def plot_decay_curve_comparison(r163: dict, r176: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data, color, marker in [
        ("exp_0163", r163, "#2196F3", "o"),
        ("exp_0176", r176, "#FF5722", "s"),
    ]:
        ticks = [s["tick"] for s in data["snapshots"]]
        recalls = [s["recall_rate"] for s in data["snapshots"]]
        ax.plot(ticks, recalls, marker=marker, label=label, color=color,
                linewidth=2, markersize=5)

    ax.set_xlabel("Tick")
    ax.set_ylabel("Recall Rate")
    ax.set_title("Decay Curve Comparison: Recall over Time")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "1_decay_curve_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Threshold Discriminability
# ---------------------------------------------------------------------------
def plot_threshold_discriminability(r163: dict, r176: dict) -> None:
    thresholds = ["0.2", "0.3", "0.4", "0.5"]
    final_163 = r163["snapshots"][-1]["threshold_metrics"]
    final_176 = r176["snapshots"][-1]["threshold_metrics"]

    recall_163 = [final_163[t]["recall_rate"] for t in thresholds]
    recall_176 = [final_176[t]["recall_rate"] for t in thresholds]

    x = np.arange(len(thresholds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, recall_163, width, label="exp_0163",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, recall_176, width, label="exp_0176",
                   color="#FF5722", alpha=0.85)

    for bar, val in zip(bars1, recall_163):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, recall_176):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Recall Rate")
    ax.set_title("Threshold Discriminability at Final Tick (t=200)")
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.legend()
    ax.set_ylim(0, 0.55)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_threshold_discriminability.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Activation Distribution (modeled)
# ---------------------------------------------------------------------------
def plot_activation_distribution() -> None:
    """Simulate 200 ticks of soft_floor_decay_step for three impact levels
    starting from activation=1.0, stability=0, lam=0.02. Plot resulting
    activation trajectories and mark threshold lines."""
    impacts = [0.0, 0.5, 1.0]
    colors = ["#E53935", "#FB8C00", "#43A047"]
    n_ticks = 200
    lam = 0.02

    fig, ax = plt.subplots(figsize=(8, 5))
    for impact, color in zip(impacts, colors):
        a = 1.0
        trajectory = [a]
        for _ in range(n_ticks):
            a = soft_floor_decay_step(a, impact, 0.0, lam=lam)
            trajectory.append(a)
        ax.plot(range(n_ticks + 1), trajectory, label=f"impact={impact}",
                color=color, linewidth=2)

    for thresh in [0.2, 0.3, 0.4, 0.5]:
        ax.axhline(thresh, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(n_ticks + 2, thresh, f"t={thresh}", va="center", fontsize=8,
                color="gray")

    ax.set_xlabel("Tick")
    ax.set_ylabel("Activation")
    ax.set_title("Modeled Activation Decay by Impact Level (stability=0)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, n_ticks + 10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3_activation_distribution.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Decay Rate Heatmap
# ---------------------------------------------------------------------------
def plot_decay_rate_heatmap() -> None:
    lam = 0.02
    n = 100
    impacts = np.linspace(0, 1, n)
    activations = np.linspace(0.01, 1.0, n)
    I, A = np.meshgrid(impacts, activations)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, stab, title_suffix in zip(
        axes, [0.0, 0.5], ["stability=0", "stability=0.5"]
    ):
        decay_rates = np.zeros_like(I)
        for i in range(n):
            for j in range(n):
                a_t = float(A[i, j])
                imp = float(I[i, j])
                a_next = soft_floor_decay_step(a_t, imp, stab, lam=lam)
                if a_t > 1e-9:
                    decay_rates[i, j] = -(a_next - a_t) / a_t
                else:
                    decay_rates[i, j] = 0.0

        im = ax.pcolormesh(I, A, decay_rates, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label="Decay Rate")
        ax.set_xlabel("Impact")
        ax.set_ylabel("Activation")
        ax.set_title(f"Decay Rate Heatmap ({title_suffix})")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "4_decay_rate_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Score Decomposition
# ---------------------------------------------------------------------------
def plot_score_decomposition(r163: dict, r176: dict, r000: dict) -> None:
    labels = ["exp_0000", "exp_0163", "exp_0176"]
    data_map = {"exp_0000": r000, "exp_0163": r163, "exp_0176": r176}
    metrics = ["retrieval_score", "plausibility_score", "overall_score"]
    metric_labels = ["Retrieval", "Plausibility", "Overall"]
    colors = ["#42A5F5", "#66BB6A", "#FFA726"]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, mlabel, color) in enumerate(
        zip(metrics, metric_labels, colors)
    ):
        vals = [data_map[l][metric] for l in labels]
        bars = ax.bar(x + i * width, vals, width, label=mlabel, color=color,
                      alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_title("Score Decomposition: Retrieval vs Plausibility vs Overall")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 0.85)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "5_score_decomposition.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. MRR vs Precision_Strict over Ticks
# ---------------------------------------------------------------------------
def plot_mrr_vs_precision(r176: dict) -> None:
    ticks = [s["tick"] for s in r176["snapshots"]]
    mrr_vals = [s["mrr_mean"] for s in r176["snapshots"]]
    prec_vals = [s["precision_strict"] for s in r176["snapshots"]]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#1565C0"
    color2 = "#C62828"

    ax1.plot(ticks, mrr_vals, marker="o", color=color1, linewidth=2,
             markersize=5, label="MRR Mean")
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("MRR Mean", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(ticks, prec_vals, marker="s", color=color2, linewidth=2,
             markersize=5, label="Precision Strict")
    ax2.set_ylabel("Precision Strict", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("MRR vs Precision Strict over Time (exp_0176)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "6_mrr_vs_precision.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    r163 = load_results("exp_0163")
    r176 = load_results("exp_0176")
    r000 = load_results("exp_0000")

    plot_decay_curve_comparison(r163, r176)
    plot_threshold_discriminability(r163, r176)
    plot_activation_distribution()
    plot_decay_rate_heatmap()
    plot_score_decomposition(r163, r176, r000)
    plot_mrr_vs_precision(r176)

    n = len(list(OUT_DIR.glob("*.png")))
    print(f"Generated {n} figures in docs/figures/analysis/")


if __name__ == "__main__":
    main()
