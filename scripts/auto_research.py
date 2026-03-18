"""Auto-research loop: sweep soft_floor_decay_step parameters."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
HISTORY = EXP_DIR / "history.jsonl"
BEST_DIR = EXP_DIR / "best"

# Base params from exp_0176
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

# 20 experiment configs: (param_overrides, hypothesis)
EXPERIMENTS = [
    # Phase 1: floor_max sweep
    ({"floor_max": 0.45}, "Raise floor_max to 0.45 — higher floor retains more recall while keeping some threshold discrimination."),
    ({"floor_max": 0.50}, "floor_max=0.50 — midpoint between 0.35 (too low recall) and 0.79 (degenerate)."),
    ({"floor_max": 0.55}, "floor_max=0.55 — closer to upper range, test recall/discrimination tradeoff."),
    ({"floor_max": 0.60}, "floor_max=0.60 — near upper bound, check if discrimination still exists."),
    ({"floor_max": 0.65}, "floor_max=0.65 — approaching exp_0163 territory, check if discrimination survives."),
    # Phase 2: floor_power sweep (with best floor_max from phase 1, defaulting to 0.50)
    ({"floor_max": 0.50, "floor_power": 1.0}, "Linear floor scaling (power=1.0) with floor_max=0.50 — more even spread across impact values."),
    ({"floor_max": 0.50, "floor_power": 1.5}, "Sub-quadratic floor scaling (power=1.5) with floor_max=0.50."),
    ({"floor_max": 0.50, "floor_power": 3.0}, "Cubic floor scaling (power=3.0) — concentrates floor boost on highest-impact memories."),
    # Phase 3: consolidation_gain sweep
    ({"floor_max": 0.50, "consolidation_gain": 0.3}, "Lower consolidation_gain=0.3 — less gate-dependent decay slowing."),
    ({"floor_max": 0.50, "consolidation_gain": 0.8}, "Higher consolidation_gain=0.8 — stronger consolidation for high-impact memories."),
    ({"floor_max": 0.50, "consolidation_gain": 0.9}, "consolidation_gain=0.9 — near-maximal consolidation effect."),
    # Phase 4: gate params
    ({"floor_max": 0.50, "gate_center": 0.3, "gate_width": 0.06}, "Lower gate_center=0.3 — consolidation kicks in at lower activation."),
    ({"floor_max": 0.50, "gate_center": 0.5, "gate_width": 0.10}, "Higher gate_center=0.5, wider gate — broader consolidation zone."),
    # Phase 5: lambda tuning
    ({"floor_max": 0.50, "lambda_fact": 0.008, "lambda_episode": 0.024}, "Slower decay rates — lambda_fact=0.008, lambda_episode=0.024."),
    ({"floor_max": 0.50, "lambda_fact": 0.015, "lambda_episode": 0.045}, "Faster decay rates — lambda_fact=0.015, lambda_episode=0.045."),
    # Phase 6: alpha and floor_min
    ({"floor_max": 0.50, "alpha": 2.5}, "Higher alpha=2.5 — stronger exponential impact modifier."),
    ({"floor_max": 0.50, "alpha": 1.5}, "Lower alpha=1.5 — weaker exponential impact modifier."),
    ({"floor_max": 0.50, "floor_min": 0.10}, "Higher floor_min=0.10 — raises minimum floor for low-impact memories."),
    ({"floor_max": 0.50, "floor_min": 0.02, "floor_max": 0.55}, "floor_min=0.02, floor_max=0.55 — wider floor range."),
    # Phase 7: min_rate_scale
    ({"floor_max": 0.50, "min_rate_scale": 0.05}, "Lower min_rate_scale=0.05 — allows deeper rate reduction for consolidated memories."),
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
            data = json.load(f)
        return data
    else:
        print(f"  FAILED: {result.stderr[-200:]}")
        return None


def main():
    best_score = get_best_score()
    print(f"Current best: {best_score:.4f}")

    start_num = 177
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

        status = "improved" if overall > best_score else "no_gain"
        delta = overall - best_score

        print(f"  overall={overall:.4f} retrieval={retrieval:.4f} plausibility={plausibility:.4f}")
        print(f"  recall_mean={recall_mean:.4f} thresh_disc={thresh_disc:.4f}")
        print(f"  delta={delta:+.4f} → {status}")

        record = {
            "experiment": exp_name,
            "overall_score": round(overall, 4),
            "retrieval_score": round(retrieval, 4),
            "plausibility_score": round(plausibility, 4),
            "recall_mean": round(recall_mean, 4),
            "threshold_discrimination": round(thresh_disc, 4),
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
            # Update best symlink
            best_link = EXP_DIR / "best"
            if best_link.is_symlink():
                best_link.unlink()
            elif best_link.exists():
                shutil.rmtree(best_link)
            best_link.symlink_to(exp_name)
            print(f"  *** NEW BEST: {exp_name} ({overall:.4f}) ***")

            # Update BASE_PARAMS for subsequent experiments
            params = {**BASE_PARAMS, **overrides}
            for key, val in overrides.items():
                BASE_PARAMS[key] = val

    print(f"\n{'='*60}")
    print(f"Auto-research complete. {improvements} improvements found.")
    print(f"Final best: {best_score:.4f}")


if __name__ == "__main__":
    main()
