"""Generate node-level memory lifecycle visualizations.

This script runs a focused simulation to track the state of specific memory nodes
over time. It generates 4 distinct plots:
1. Dual-axis: Activation vs Stability (lifecycle of mem_0001)
2. Impact ablation: High vs Low impact decay
3. Type ablation: Fact vs Episode decay
4. Cascade effect: Network activation spike on query

It DOES NOT affect the main research iteration or evaluator pipeline.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from memory_decay.main import build_graph_from_dataset
from memory_decay.decay import DecayEngine

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


def load_dataset(path: str) -> list[dict]:
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def run_tracking_simulation(dataset: list[dict], total_ticks: int = 200):
    """Run simulation and record states of target nodes."""
    print("Building graph for tracing...")
    graph = build_graph_from_dataset(dataset, embedding_backend="local")
    
    # We will use power_law, as experiments found cubic is the best shape
    engine = DecayEngine(graph, decay_type="power_law")
    
    # Define interesting nodes to track
    # 1. Dual axis target (will be heavily rehearsed)
    node_dual = "mem_0001"  # 한글/세종대왕 (Fact)
    
    # 2. Impact comparison
    node_high_impact = "mem_0014" # 첫 차 드라이브 (Episode, impact 0.6)
    node_low_impact = "mem_0003"  # 백두산 (Fact, impact 0.3)
    
    # 3. Type comparison
    node_fact = "mem_0006"     # 서울 (Fact, impact 0.4)
    node_episode = "mem_0005"  # 오락실 (Episode, impact 0.5)
    
    # 4. Cascade source and targets
    node_cascade_src = "mem_0010" # 설날/추석 (Fact)
    node_cascade_tgt1 = "mem_0002" # 한복 (Fact)
    node_cascade_tgt2 = "mem_0008" # 비빔밥 (Fact)
    
    # Define tracking structures
    targets = [
        node_dual, node_high_impact, node_low_impact, 
        node_fact, node_episode, 
        node_cascade_src, node_cascade_tgt1, node_cascade_tgt2
    ]
    history = {t: {"activation": [], "stability": []} for t in targets}
    
    print("Running ticks...")
    for tick in range(total_ticks):
        # Trigger rehearsal for dual axis target
        if tick in [20, 50, 100]:
            p = engine.get_params()
            graph.re_activate(
                node_dual,
                boost_amount=1.0,  # Maximize activation
                current_tick=tick,
                reinforcement_gain_direct=p.get("reinforcement_gain_direct", 0.2),
                reinforcement_gain_assoc=p.get("reinforcement_gain_assoc", 0.05),
                stability_cap=p.get("stability_cap", 1.0)
            )
            
        # Trigger cascade demo
        if tick == 80:
            # Querying node_cascade_src should cause cascade to tgts
            p = engine.get_params()
            graph.re_activate(
                node_cascade_src,
                boost_amount=1.0,
                current_tick=tick,
                reinforcement_gain_direct=p.get("reinforcement_gain_direct", 0.2),
                reinforcement_gain_assoc=p.get("reinforcement_gain_assoc", 0.05),
                stability_cap=p.get("stability_cap", 1.0)
            )
        
        # Record state before tick
        for t in targets:
            node = graph.get_node(t)
            history[t]["activation"].append(node["activation_score"])
            history[t]["stability"].append(node["stability_score"])
            
        # Apply decay
        engine.tick()
        
    return history


def plot_dual_axis(history, target_id: str, out_dir: Path):
    ticks = np.arange(len(history[target_id]["activation"]))
    act = history[target_id]["activation"]
    stab = history[target_id]["stability"]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Time (Ticks)')
    ax1.set_ylabel('Activation Score', color=color1)
    ax1.plot(ticks, act, color=color1, linewidth=2, label='Activation')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Recall Threshold (0.3)')
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Stability Score', color=color2)
    ax2.plot(ticks, stab, color=color2, linewidth=2, linestyle='-.', label='Stability')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.title(f'Memory Lifecycle (Activation vs Stability)\nTarget: {target_id} (Rehearsed at t=20, 50, 100)')
    fig.tight_layout()
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center')
    
    plt.savefig(out_dir / "node_lifecycle_dual.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_impact_comparison(history, high_id: str, low_id: str, out_dir: Path):
    ticks = np.arange(len(history[high_id]["activation"]))
    
    plt.figure(figsize=(10, 5))
    plt.plot(ticks, history[high_id]["activation"], label=f'High Impact (0.6) - {high_id}', color='purple', linewidth=2)
    plt.plot(ticks, history[low_id]["activation"], label=f'Low Impact (0.3) - {low_id}', color='gray', linewidth=2)
    
    plt.axhline(y=0.3, color='black', linestyle='--', alpha=0.3)
    plt.title('Effect of Emotional Impact on Forgetting Curve')
    plt.xlabel('Time (Ticks)')
    plt.ylabel('Activation Score')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(out_dir / "node_impact_ablation.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_type_comparison(history, fact_id: str, episode_id: str, out_dir: Path):
    ticks = np.arange(len(history[fact_id]["activation"]))
    
    plt.figure(figsize=(10, 5))
    plt.plot(ticks, history[fact_id]["activation"], label=f'Semantic Fact - {fact_id}', color='green', linewidth=2)
    plt.plot(ticks, history[episode_id]["activation"], label=f'Episodic Memory - {episode_id}', color='orange', linewidth=2)
    
    plt.axhline(y=0.3, color='black', linestyle='--', alpha=0.3)
    plt.title('Semantic (Fact) vs Episodic Memory Forgetting Curves')
    plt.xlabel('Time (Ticks)')
    plt.ylabel('Activation Score')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(out_dir / "node_type_ablation.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_cascade_effect(history, src_id: str, tgt1_id: str, tgt2_id: str, out_dir: Path):
    ticks = np.arange(len(history[src_id]["activation"]))
    
    plt.figure(figsize=(10, 5))
    plt.plot(ticks, history[src_id]["activation"], label=f'Queried Target - {src_id}', color='red', linewidth=2)
    plt.plot(ticks, history[tgt1_id]["activation"], label=f'Associated Node 1 - {tgt1_id}', color='blue', linestyle='--')
    plt.plot(ticks, history[tgt2_id]["activation"], label=f'Associated Node 2 - {tgt2_id}', color='cyan', linestyle='--')
    
    # Zoom in to see the spike at t=80
    plt.title('Association Cascade: Unqueried Nodes Gaining Activation (Query at t=80)')
    plt.xlabel('Time (Ticks)')
    plt.ylabel('Activation Score')
    plt.ylim(0, 1.05)
    plt.xlim(60, 120)  # Zoom window
    
    plt.axvline(x=80, color='gray', linestyle=':', alpha=0.5, label='Query Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(out_dir / "node_cascade_effect.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    data_path = "data/memories_500.jsonl"
    out_dir = Path("docs/figures/node_traces")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {data_path}...")
    dataset = load_dataset(data_path)
    
    history = run_tracking_simulation(dataset, total_ticks=200)
    
    print(f"Generating plots in {out_dir}...")
    plot_dual_axis(history, "mem_0001", out_dir)
    plot_impact_comparison(history, "mem_0014", "mem_0003", out_dir)
    plot_type_comparison(history, "mem_0006", "mem_0005", out_dir)
    plot_cascade_effect(history, "mem_0010", "mem_0002", "mem_0008", out_dir)
    
    print("Done! You can use these plots for your blog or report.")

if __name__ == "__main__":
    main()
