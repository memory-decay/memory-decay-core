"""Run experiment across multiple seeds and compute bootstrap CIs."""

import numpy as np
from pathlib import Path
from memory_decay.runner import run_experiment


METRICS = [
    "overall_score", "retrieval_score", "plausibility_score",
    "recall_mean", "mrr_mean", "precision_lift", "precision_strict",
    "corr_score", "smoothness_score", "threshold_discrimination",
]


def run_multi_seed(
    exp_dir: Path,
    seeds: range | list[int] = range(42, 72),
    cache_dir: Path = Path("cache"),
    ci_level: float = 0.95,
) -> dict:
    """Run experiment with multiple seeds, return stats with bootstrap CIs."""
    individual = []
    for seed in seeds:
        result = run_experiment(str(exp_dir), cache_dir=str(cache_dir), seed=seed)
        individual.append(result)

    n = len(individual)
    stats = {"n_seeds": n, "individual_scores": individual, "diagnostic_only": True}

    for agg_key in ("mean", "std", "ci_lower", "ci_upper"):
        stats[agg_key] = {}

    for metric in METRICS:
        values = np.array([r.get(metric, 0.0) for r in individual])
        stats["mean"][metric] = float(np.mean(values))
        stats["std"][metric] = float(np.std(values, ddof=1)) if n > 1 else 0.0

        # Bootstrap 95% CI
        if n > 1:
            rng = np.random.default_rng(0)
            boot_means = []
            for _ in range(10000):
                sample = rng.choice(values, size=n, replace=True)
                boot_means.append(np.mean(sample))
            boot_means = np.sort(boot_means)
            alpha = (1 - ci_level) / 2
            stats["ci_lower"][metric] = float(boot_means[int(alpha * 10000)])
            stats["ci_upper"][metric] = float(boot_means[int((1 - alpha) * 10000)])
        else:
            stats["ci_lower"][metric] = stats["mean"][metric]
            stats["ci_upper"][metric] = stats["mean"][metric]

    return stats


def compare_experiments(
    exp_a: Path, exp_b: Path,
    seeds: range | list[int] = range(42, 72),
    cache_dir: Path = Path("cache"),
) -> dict:
    """Paired comparison between two experiments across same seeds."""
    diffs = {m: [] for m in METRICS}
    for seed in seeds:
        res_a = run_experiment(str(exp_a), cache_dir=str(cache_dir), seed=seed)
        res_b = run_experiment(str(exp_b), cache_dir=str(cache_dir), seed=seed)
        for m in METRICS:
            diffs[m].append(res_b.get(m, 0.0) - res_a.get(m, 0.0))

    from scipy import stats as sp_stats
    results = {}
    for m in METRICS:
        d = np.array(diffs[m])
        t_stat, p_val = sp_stats.ttest_1samp(d, 0.0) if len(d) > 1 else (0.0, 1.0)
        results[m] = {
            "mean_diff": float(np.mean(d)),
            "std_diff": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant_005": bool(p_val < 0.05),
        }
    return results
