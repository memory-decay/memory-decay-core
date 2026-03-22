"""Run cross-encoder sweep experiments: 0452-0455.

0452: 0292 base + CE=0.2 (no BM25)
0453: 0292 base + BM25=0.3 + CE=0.2
0454: 0292 base + CE=0.3 (no BM25)
0455: 0292 base + BM25=0.3 + CE=0.3
"""
from pathlib import Path
from src.memory_decay.runner import run_experiment
import json
import sys

EXPS = [
    "experiments/exp_lme_0452",
    "experiments/exp_lme_0453",
    "experiments/exp_lme_0454",
    "experiments/exp_lme_0455",
]

for exp_dir in EXPS:
    exp_path = Path(exp_dir)
    print(f"\n{'='*60}")
    print(f"Running {exp_path.name}...")
    with open(exp_path / "hypothesis.txt") as f:
        print(f.read().strip())
    print()

    try:
        result = run_experiment(
            experiment_dir=str(exp_path),
            cache_dir="cache",
            total_ticks=200,
            eval_interval=20,
            reactivation_policy="scheduled_query",
            force=True,
        )
        m = result.get("metrics", result)
        overall = m.get("overall_score", 0)
        recall = m.get("similarity_recall_rate", 0)
        ps = m.get("precision_strict", 0)
        pl = m.get("precision_lift", 0)
        plaus = m.get("plausibility_score", 0)
        print(f"  overall={overall:.4f}  recall={recall:.4f}  precision_strict={ps:.4f}")
        print(f"  precision_lift={pl:.4f}  plausibility={plaus:.4f}")

        with open(exp_path / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {exp_path}/results.json")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n\nSummary:")
print(f"{'exp':<25} {'overall':>8} {'recall':>8} {'ps':>8} {'plaus':>8}")
for exp_dir in EXPS:
    rpath = Path(exp_dir) / "results.json"
    if rpath.exists():
        with open(rpath) as f:
            r = json.load(f)
        m = r.get("metrics", r)
        print(f"{Path(exp_dir).name:<25} {m.get('overall_score',0):>8.4f} {m.get('similarity_recall_rate',0):>8.4f} {m.get('precision_strict',0):>8.4f} {m.get('plausibility_score',0):>8.4f}")
