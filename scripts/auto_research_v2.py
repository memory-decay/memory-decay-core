"""Auto-research loop v2: high floor_max + slow lambda strategies."""

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
HISTORY = EXP_DIR / "history.jsonl"
BEST_DIR = EXP_DIR / "best"

BASE_PARAMS = {
    "lambda_fact": 0.012, "lambda_episode": 0.036,
    "alpha": 2.0, "stability_weight": 0.8,
    "stability_decay": 0.01, "reinforcement_gain_direct": 0.2,
    "reinforcement_gain_assoc": 0.05, "stability_cap": 1.0,
    "floor_min": 0.05, "floor_max": 0.35, "floor_power": 2.0,
    "gate_center": 0.4, "gate_width": 0.08,
    "consolidation_gain": 0.6, "min_rate_scale": 0.1,
}

DECAY_FN_CODE = '''"""Use soft_floor_decay_step from decay.py."""

from memory_decay.decay import soft_floor_decay_step


def compute_decay(activation, impact, stability, mtype, params):
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)
    return soft_floor_decay_step(
        activation,
        impact,
        stability,
        lam=lam,
        alpha=params.get("alpha", 2.0),
        rho=params.get("stability_weight", 0.8),
        floor_min=params.get("floor_min", 0.05),
        floor_max=params.get("floor_max", 0.35),
        floor_power=params.get("floor_power", 2.0),
        gate_center=params.get("gate_center", 0.4),
        gate_width=params.get("gate_width", 0.08),
        consolidation_gain=params.get("consolidation_gain", 0.6),
        min_rate_scale=params.get("min_rate_scale", 0.1),
    )
'''

EXPERIMENTS = [
    # Strategy 1: Very high floor_max with low power — keep high-impact memories alive
    ({"floor_max": 0.75, "floor_power": 1.0, "floor_min": 0.10},
     "floor_max=0.75 power=1.0 min=0.10 — high-impact floors near 0.75 maintain recall, low-impact floor=0.10 allows discrimination."),
    ({"floor_max": 0.75, "floor_power": 0.5, "floor_min": 0.10},
     "floor_max=0.75 power=0.5 — sqrt scaling concentrates most memories near high floor, only very low impact decays."),
    ({"floor_max": 0.78, "floor_power": 1.0, "floor_min": 0.15},
     "floor_max=0.78 power=1.0 min=0.15 — near-degenerate but with linear floor spread."),

    # Strategy 2: High floor + slow lambda + high consolidation
    ({"floor_max": 0.75, "floor_power": 1.0, "floor_min": 0.10, "lambda_fact": 0.006, "lambda_episode": 0.018},
     "High floor + very slow lambda (0.006/0.018) — slow convergence keeps activation spread."),
    ({"floor_max": 0.75, "floor_power": 1.0, "floor_min": 0.10, "consolidation_gain": 0.9},
     "High floor + max consolidation_gain=0.9 — strong retention for high-activation high-impact memories."),
    ({"floor_max": 0.75, "floor_power": 1.0, "floor_min": 0.10, "consolidation_gain": 0.9, "lambda_fact": 0.008, "lambda_episode": 0.024},
     "High floor + slow lambda + max consolidation — maximize retention and ranking quality."),

    # Strategy 3: Aggressive consolidation with moderate floor
    ({"floor_max": 0.65, "floor_power": 0.5, "floor_min": 0.15, "consolidation_gain": 0.9, "gate_center": 0.3, "gate_width": 0.05},
     "Moderate floor + aggressive early consolidation (gate_center=0.3) — lock in high-impact memories early."),
    ({"floor_max": 0.70, "floor_power": 0.5, "floor_min": 0.10, "consolidation_gain": 0.9, "alpha": 2.5},
     "floor_max=0.70 sqrt + strong consolidation + high alpha — maximize impact differentiation."),

    # Strategy 4: Fine-tune around best from v1 (exp_0190: floor_max=0.50, slow lambda)
    ({"floor_max": 0.50, "floor_power": 1.0, "lambda_fact": 0.006, "lambda_episode": 0.018, "consolidation_gain": 0.9},
     "exp_0190 base + power=1.0 + max consolidation — fix MRR by increasing retention."),
    ({"floor_max": 0.50, "floor_power": 1.0, "lambda_fact": 0.004, "lambda_episode": 0.012, "consolidation_gain": 0.9},
     "Ultra-slow lambda (0.004/0.012) + max consolidation — push recall near degenerate."),

    # Strategy 5: Extreme alpha to boost impact-differentiation
    ({"floor_max": 0.75, "floor_power": 1.0, "floor_min": 0.10, "alpha": 3.0, "consolidation_gain": 0.8},
     "alpha=3.0 — very strong exponential impact modifier, high floor, strong consolidation."),
    ({"floor_max": 0.75, "floor_power": 1.0, "floor_min": 0.10, "alpha": 4.0, "consolidation_gain": 0.8},
     "alpha=4.0 — extreme exponential impact modifier for maximum activation spread."),

    # Strategy 6: Wide gate to extend consolidation across more activation range
    ({"floor_max": 0.75, "floor_power": 1.0, "floor_min": 0.10, "consolidation_gain": 0.9, "gate_center": 0.5, "gate_width": 0.20},
     "Very wide gate (width=0.20, center=0.5) — consolidation active across broad activation range."),
    ({"floor_max": 0.75, "floor_power": 1.0, "floor_min": 0.10, "consolidation_gain": 0.9, "gate_center": 0.6, "gate_width": 0.15},
     "High gate_center=0.6 — consolidation primarily protects high-activation memories."),

    # Strategy 7: Combine best ideas
    ({"floor_max": 0.75, "floor_power": 0.5, "floor_min": 0.10, "alpha": 3.0, "consolidation_gain": 0.9, "lambda_fact": 0.008, "lambda_episode": 0.024},
     "Kitchen sink: high floor, sqrt power, alpha=3, max consolidation, slow lambda."),
    ({"floor_max": 0.78, "floor_power": 0.5, "floor_min": 0.15, "alpha": 3.0, "consolidation_gain": 0.9, "lambda_fact": 0.006, "lambda_episode": 0.018, "gate_center": 0.5, "gate_width": 0.15},
     "Max everything: floor=0.78, sqrt, alpha=3, consolidation=0.9, very slow lambda, wide gate."),

    # Strategy 8: Try near-zero floor_min with very high floor_max
    ({"floor_max": 0.78, "floor_power": 1.0, "floor_min": 0.01},
     "Widest range: floor_min=0.01 to floor_max=0.78 linear — maximum discrimination potential."),
    ({"floor_max": 0.80, "floor_power": 1.0, "floor_min": 0.05},
     "floor_max=0.80 — test if at this level we still get discrimination or if it's degenerate."),

    # Strategy 9: Ultra-low min_rate_scale
    ({"floor_max": 0.75, "floor_power": 1.0, "floor_min": 0.10, "consolidation_gain": 0.95, "min_rate_scale": 0.01},
     "Near-zero rate for consolidated memories — almost freeze high-impact high-activation memories."),
    ({"floor_max": 0.75, "floor_power": 1.0, "floor_min": 0.10, "consolidation_gain": 0.9, "min_rate_scale": 0.02, "alpha": 3.0},
     "min_rate_scale=0.02 + alpha=3 + consolidation=0.9 — near-frozen consolidated memories with strong impact."),
]


def get_best_score():
    best_results = BEST_DIR / "results.json"
    if best_results.exists():
        with open(best_results) as f:
            return json.load(f).get("overall_score", 0)
    return 0


def run_experiment(exp_num, param_overrides, hypothesis):
    exp_name = f"exp_{exp_num:04d}"
    exp_path = EXP_DIR / exp_name
    exp_path.mkdir(exist_ok=True)

    params = {**BASE_PARAMS, **param_overrides}
    with open(exp_path / "params.json", "w") as f:
        json.dump(params, f)
    with open(exp_path / "decay_fn.py", "w") as f:
        f.write(DECAY_FN_CODE)
    with open(exp_path / "hypothesis.txt", "w") as f:
        f.write(hypothesis + "\n")

    result = subprocess.run(
        ["uv", "run", "python", "-m", "memory_decay.runner", str(exp_path), "--cache", "cache"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=120,
    )

    results_file = exp_path / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    else:
        print(f"  FAILED: {result.stderr[-200:]}")
        return None


def main():
    best_score = get_best_score()
    print(f"Current best: {best_score:.4f}")

    start_num = 197
    improvements = 0

    for i, (overrides, hypothesis) in enumerate(EXPERIMENTS):
        exp_num = start_num + i
        exp_name = f"exp_{exp_num:04d}"
        print(f"\n{'='*60}")
        print(f"[{i+1}/20] {exp_name}: {hypothesis[:80]}")
        print(f"  Overrides: {overrides}")

        result = run_experiment(exp_num, overrides, hypothesis)
        if result is None:
            record = {"experiment": exp_name, "status": "failed", "hypothesis": hypothesis}
            with open(HISTORY, "a") as f:
                f.write(json.dumps(record) + "\n")
            continue

        overall = result.get("overall_score", 0)
        retrieval = result.get("retrieval_score", 0)
        plausibility = result.get("plausibility_score", 0)
        recall_mean = result.get("recall_mean", 0)
        thresh_disc = result.get("threshold_discrimination", 0)
        mrr = result.get("mrr_mean", 0)
        corr = result.get("corr_score", 0)

        status = "improved" if overall > best_score else "no_gain"
        delta = overall - best_score

        print(f"  overall={overall:.4f} retrieval={retrieval:.4f} plausibility={plausibility:.4f}")
        print(f"  recall={recall_mean:.4f} mrr={mrr:.4f} thresh_disc={thresh_disc:.4f} corr={corr:.4f}")
        print(f"  delta={delta:+.4f} → {status}")

        record = {
            "experiment": exp_name,
            "overall_score": round(overall, 4),
            "retrieval_score": round(retrieval, 4),
            "plausibility_score": round(plausibility, 4),
            "recall_mean": round(recall_mean, 4),
            "mrr_mean": round(mrr, 4),
            "threshold_discrimination": round(thresh_disc, 4),
            "corr_score": round(corr, 4),
            "status": status,
            "delta": round(delta, 4),
            "overrides": overrides,
            "hypothesis": hypothesis,
        }
        with open(HISTORY, "a") as f:
            f.write(json.dumps(record) + "\n")

        if status == "improved":
            best_score = overall
            improvements += 1
            best_link = EXP_DIR / "best"
            if best_link.is_symlink():
                best_link.unlink()
            elif best_link.exists():
                shutil.rmtree(best_link)
            best_link.symlink_to(exp_name)
            print(f"  *** NEW BEST: {exp_name} ({overall:.4f}) ***")

    print(f"\n{'='*60}")
    print(f"Auto-research v2 complete. {improvements} improvements found.")
    print(f"Final best: {best_score:.4f}")


if __name__ == "__main__":
    main()
