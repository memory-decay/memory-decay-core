"""Strict evaluation wrapper — prevents decay function gaming.

Anti-gaming measures:
1. HARD CONSTRAINT: impact=0 items must decay below 0.3 (fact) / 0.25 (episode) by tick 200
2. FORGETTING QUALITY: measures actual forgetting depth, replaces smoothness reward for flat curves
3. ADJUSTED SCORING: plausibility = 0.4*corr + 0.3*forgetting_quality + 0.3*smoothness
   (but smoothness is zeroed if forgetting_depth < 5%)

Usage:
    PYTHONPATH=src uv run python experiments/strict_eval.py experiments/exp_NNNN --cache cache
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory_decay.runner import run_experiment, validate_decay_fn


def load_decay_fn(fn_path: str):
    spec = importlib.util.spec_from_file_location("decay_fn", fn_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_decay


def strict_validate(fn_path: str, params: dict) -> tuple[bool, str | None]:
    """Extended validation: simulate 200 ticks and check minimum decay."""

    # First pass standard validation
    ok, error = validate_decay_fn(fn_path, params)
    if not ok:
        return False, error

    fn = load_decay_fn(fn_path)

    # Simulate 200 ticks for impact=0, stability=0 (no protection)
    a_fact = 0.5
    a_episode = 0.5
    stability = 0.0
    stability_decay = params.get("stability_decay", 0.01)

    for _ in range(200):
        a_fact = max(min(fn(a_fact, 0.0, stability, "fact", params), 1.0), 0.0)
        a_episode = max(min(fn(a_episode, 0.0, stability, "episode", params), 1.0), 0.0)
        stability = max(0.0, stability * (1.0 - stability_decay))

    if a_fact > 0.30:
        return False, (
            f"Insufficient decay: fact(impact=0, stability=0) after 200 ticks = {a_fact:.4f} > 0.30. "
            f"Low-impact items must actually be forgotten."
        )

    if a_episode > 0.25:
        return False, (
            f"Insufficient decay: episode(impact=0, stability=0) after 200 ticks = {a_episode:.4f} > 0.25. "
            f"Low-impact episodes must decay faster than facts."
        )

    # Also check a mid-impact item doesn't stay at ceiling
    a_mid = 0.5
    stability = 0.0
    for _ in range(200):
        a_mid = max(min(fn(a_mid, 0.3, stability, "fact", params), 1.0), 0.0)
        stability = max(0.0, stability * (1.0 - stability_decay))

    if a_mid > 0.45:
        return False, (
            f"Insufficient decay: fact(impact=0.3) after 200 ticks = {a_mid:.4f} > 0.45. "
            f"Moderate-impact items should show meaningful decay."
        )

    return True, None


def compute_strict_score(result: dict) -> dict:
    """Compute adjusted score with anti-gaming metrics."""
    if result["status"] != "completed":
        return result

    snapshots = result["snapshots"]

    # Forgetting depth: how much recall dropped from tick 0 to tick 200
    recall_0 = snapshots[0]["recall_mean"]
    recall_final = snapshots[-1]["recall_mean"]
    forgetting_depth = max(0.0, 1.0 - recall_final / max(recall_0, 1e-9))

    # Forgetting quality: want 20-50% forgetting for full score
    # 0% forgetting → 0.0, 30%+ → 1.0
    forgetting_score = min(forgetting_depth / 0.30, 1.0)

    # Adjusted smoothness: flat curves (< 5% forgetting) get 0 smoothness
    original_smoothness = result["smoothness_score"]
    if forgetting_depth < 0.05:
        adj_smoothness = 0.0
    else:
        adj_smoothness = original_smoothness

    # Correlation (unchanged)
    corr_score = result["corr_score"]

    # Adjusted plausibility: 40% correlation, 30% forgetting quality, 30% smoothness
    adj_plausibility = 0.4 * corr_score + 0.3 * forgetting_score + 0.3 * adj_smoothness

    # Overall: same 70/30 split
    retrieval_score = result["retrieval_score"]
    adj_overall = 0.7 * retrieval_score + 0.3 * adj_plausibility

    result["forgetting_depth"] = round(forgetting_depth, 4)
    result["forgetting_score"] = round(forgetting_score, 4)
    result["adj_smoothness"] = round(adj_smoothness, 4)
    result["adj_plausibility"] = round(adj_plausibility, 4)
    result["strict_score"] = round(adj_overall, 6)

    result["precision_strict"] = result.get("precision_strict", 0.0)
    result["precision_associative"] = result.get("precision_associative", 0.0)
    result["similarity_recall_rate"] = result.get("similarity_recall_rate", 0.0)

    return result


def run_strict_experiment(experiment_dir: str, cache_dir: str) -> dict:
    """Run experiment with strict anti-gaming validation and adjusted scoring."""
    exp_path = Path(experiment_dir)

    with open(exp_path / "params.json", "r") as f:
        params = json.load(f)

    fn_path = str(exp_path / "decay_fn.py")

    # Strict validation first
    ok, error = strict_validate(fn_path, params)
    if not ok:
        result = {"status": "validation_failed", "error": error}
        with open(exp_path / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    # Standard experiment
    result = run_experiment(experiment_dir, cache_dir)

    # Compute strict score
    result = compute_strict_score(result)

    # Save updated results
    with open(exp_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Strict decay experiment runner")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--cache", default="cache", help="Cache directory")
    args = parser.parse_args()

    result = run_strict_experiment(args.experiment_dir, args.cache)

    status = result["status"]
    if status == "completed":
        print(f"\nStrict: strict_score={result['strict_score']:.4f} "
              f"overall={result['overall_score']:.4f} "
              f"forgetting={result['forgetting_depth']:.1%} "
              f"corr={result['corr_score']:.3f} "
              f"recall={result['recall_mean']:.3f}")
    else:
        print(f"Failed: {result.get('error', 'unknown')}")
