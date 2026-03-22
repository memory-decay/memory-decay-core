"""CE weight sweep: 0456-0459 (CE=0.05, 0.1, 0.15, 0.2 on 0292 base)."""
from pathlib import Path
from src.memory_decay.runner import run_experiment
import json

EXPS = [
    "experiments/exp_lme_0452",  # control
    "experiments/exp_lme_0456",  # CE=0.05
    "experiments/exp_lme_0457",  # CE=0.1
    "experiments/exp_lme_0458",  # CE=0.15
    "experiments/exp_lme_0459",  # CE=0.2
    "experiments/exp_lme_0454",  # CE=0.3 (reference)
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
        plaus = m.get("plausibility_score", 0)
        print(f"  overall={overall:.4f}  recall={recall:.4f}  precision_strict={ps:.4f}")
        print(f"  plausibility={plaus:.4f}")

        with open(exp_path / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {exp_path}/results.json")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n\nSummary — CE weight sweep:")
print(f"{'exp':<25} {'ce_w':>6} {'overall':>8} {'recall':>8} {'plaus':>8}")
for exp_dir in EXPS:
    rpath = Path(exp_dir) / "results.json"
    ppath = Path(exp_dir) / "params.json"
    if rpath.exists():
        with open(rpath) as f:
            r = json.load(f)
        with open(ppath) as f:
            p = json.load(f)
        m = r.get("metrics", r)
        ce_w = p.get("cross_encoder_weight", 0.0)
        print(f"{Path(exp_dir).name:<25} {ce_w:>6.2f} {m.get('overall_score',0):>8.4f} {m.get('similarity_recall_rate',0):>8.4f} {m.get('plausibility_score',0):>8.4f}")
