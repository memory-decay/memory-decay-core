"""Validate key experiments across 30 seeds."""
import json
from pathlib import Path
from memory_decay.multi_runner import run_multi_seed, compare_experiments

EXPERIMENTS = {
    "baseline": Path("experiments/exp_0000"),
    "jost_p4": Path("experiments/exp_0315"),
    "best": Path("experiments/exp_0338"),
}
SEEDS = range(42, 72)  # 30 seeds
OUT = Path("experiments/multi_seed_results")


def main():
    OUT.mkdir(exist_ok=True)

    # 1. Individual multi-seed stats
    for name, exp_dir in EXPERIMENTS.items():
        print(f"\n=== {name} ({exp_dir}) ===")
        stats = run_multi_seed(exp_dir, seeds=SEEDS)
        out_file = OUT / f"{name}_stats.json"
        # Remove individual_scores for concise output
        summary = {k: v for k, v in stats.items() if k != "individual_scores"}
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        m = stats["mean"]
        ci = (stats["ci_lower"]["overall_score"], stats["ci_upper"]["overall_score"])
        print(f"  overall: {m['overall_score']:.4f} +/- {stats['std']['overall_score']:.4f}")
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # 2. Paired comparisons
    pairs = [
        ("baseline", "jost_p4"),
        ("jost_p4", "best"),
        ("baseline", "best"),
    ]
    for a_name, b_name in pairs:
        print(f"\n=== {a_name} vs {b_name} ===")
        comp = compare_experiments(
            EXPERIMENTS[a_name], EXPERIMENTS[b_name], seeds=SEEDS
        )
        out_file = OUT / f"compare_{a_name}_vs_{b_name}.json"
        with open(out_file, "w") as f:
            json.dump(comp, f, indent=2)
        ov = comp["overall_score"]
        sig = (
            "***" if ov["p_value"] < 0.001
            else "**" if ov["p_value"] < 0.01
            else "*" if ov["p_value"] < 0.05
            else "n.s."
        )
        print(f"  overall diff: {ov['mean_diff']:+.4f} (p={ov['p_value']:.4f}) {sig}")


if __name__ == "__main__":
    main()
