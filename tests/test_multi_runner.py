import json
import tempfile
from pathlib import Path

import pytest

from memory_decay.multi_runner import run_multi_seed


def test_multi_seed_returns_stats(monkeypatch):
    """Multi-seed runner must return mean, std, CI for each metric."""
    monkeypatch.setattr(
        "memory_decay.multi_runner.run_experiment",
        lambda *args, **kwargs: {"overall_score": 0.2, "retrieval_score": 0.1, "plausibility_score": 0.5, "recall_mean": 0.2, "mrr_mean": 0.1, "precision_lift": 0.0, "precision_strict": 0.05, "corr_score": 0.2, "smoothness_score": 0.9, "threshold_discrimination": 0.0},
    )
    exp_dir = Path("experiments/exp_0338")
    result = run_multi_seed(exp_dir, seeds=range(42, 45), cache_dir=Path("cache"))
    assert "mean" in result
    assert "std" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "n_seeds" in result
    assert "diagnostic_only" in result
    assert result["n_seeds"] == 3
    assert 0 < result["mean"]["overall_score"] < 1
    assert result["diagnostic_only"] is True


def test_multi_seed_different_seeds_differ(monkeypatch):
    """Different seeds should produce different individual scores."""
    monkeypatch.setattr(
        "memory_decay.multi_runner.run_experiment",
        lambda *args, seed=42, **kwargs: {"overall_score": 0.2 + (seed % 2) * 0.01, "retrieval_score": 0.1, "plausibility_score": 0.5, "recall_mean": 0.2, "mrr_mean": 0.1, "precision_lift": 0.0, "precision_strict": 0.05, "corr_score": 0.2, "smoothness_score": 0.9, "threshold_discrimination": 0.0},
    )
    exp_dir = Path("experiments/exp_0338")
    result = run_multi_seed(exp_dir, seeds=[42, 99], cache_dir=Path("cache"))
    assert result["std"]["overall_score"] >= 0
    assert len(result["individual_scores"]) == 2
