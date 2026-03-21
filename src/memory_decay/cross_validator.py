"""K-fold cross-validation for decay function evaluation."""

from __future__ import annotations

import random

import numpy as np
from pathlib import Path

METRICS = [
    "overall_score",
    "retrieval_score",
    "plausibility_score",
    "recall_mean",
    "mrr_mean",
    "corr_score",
    "retention_auc",
    "selectivity_score",
    "robustness_score",
    "eval_v2_score",
]


def run_kfold(
    exp_dir: Path,
    k: int = 5,
    cache_dir: Path = Path("cache"),
    seed: int = 42,
) -> dict:
    """Run k-fold cross-validation, rebuilding data splits each fold.

    Stratifies by memory type (fact / episode) so each fold has a
    representative mix.  Returns per-fold scores plus mean/std summaries.
    """
    import json as _json
    from memory_decay.cache_builder import load_raw_dataset
    from memory_decay.runner import run_experiment_with_split

    # Detect reactivation policy from experiment params
    params_path = exp_dir / "params.json"
    with open(params_path) as _f:
        exp_params = _json.load(_f)
    reactivation_policy = (
        "retrieval_consolidation"
        if "retrieval_consolidation_mode" in exp_params
        else "scheduled_query"
    )

    dataset = load_raw_dataset(cache_dir / "dataset.json")
    rng = random.Random(seed)

    facts = [d for d in dataset if d.get("type") == "fact"]
    episodes = [d for d in dataset if d.get("type") == "episode"]
    rng.shuffle(facts)
    rng.shuffle(episodes)

    def split_into_folds(items: list[dict], k: int) -> list[list[dict]]:
        folds: list[list[dict]] = [[] for _ in range(k)]
        for i, item in enumerate(items):
            folds[i % k].append(item)
        return folds

    fact_folds = split_into_folds(facts, k)
    episode_folds = split_into_folds(episodes, k)

    fold_scores: list[dict] = []
    for fold_idx in range(k):
        test_items = fact_folds[fold_idx] + episode_folds[fold_idx]
        train_items: list[dict] = []
        for j in range(k):
            if j != fold_idx:
                train_items.extend(fact_folds[j])
                train_items.extend(episode_folds[j])

        result = run_experiment_with_split(
            str(exp_dir),
            train_items,
            test_items,
            cache_dir=str(cache_dir),
            reactivation_policy=reactivation_policy,
            seed=seed,
        )
        fold_scores.append(result)

    stats: dict = {"fold_scores": fold_scores, "k": k, "mean": {}, "std": {}}
    for m in METRICS:
        vals = [f.get(m, 0.0) for f in fold_scores]
        stats["mean"][m] = float(np.mean(vals))
        stats["std"][m] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    key_metric = "eval_v2_score" if any("eval_v2_score" in f for f in fold_scores) else "overall_score"
    stats["worst_fold"] = min(
        fold_scores,
        key=lambda fold: fold.get(key_metric, 0.0),
    ) if fold_scores else {}

    mean_score = stats["mean"].get(key_metric, 0.0)
    stats["fold_deltas"] = [
        float(fold.get(key_metric, 0.0) - mean_score)
        for fold in fold_scores
    ]

    return stats
