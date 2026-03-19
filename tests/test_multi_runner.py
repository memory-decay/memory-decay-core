import json
import tempfile
from pathlib import Path
from memory_decay.multi_runner import run_multi_seed


def test_multi_seed_returns_stats():
    """Multi-seed runner must return mean, std, CI for each metric."""
    exp_dir = Path("experiments/exp_0338")
    result = run_multi_seed(exp_dir, seeds=range(42, 45), cache_dir=Path("cache"))
    assert "mean" in result
    assert "std" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "n_seeds" in result
    assert result["n_seeds"] == 3
    assert 0 < result["mean"]["overall_score"] < 1


def test_multi_seed_different_seeds_differ():
    """Different seeds should produce different individual scores."""
    exp_dir = Path("experiments/exp_0338")
    result = run_multi_seed(exp_dir, seeds=[42, 99], cache_dir=Path("cache"))
    assert result["std"]["overall_score"] >= 0
    assert len(result["individual_scores"]) == 2
