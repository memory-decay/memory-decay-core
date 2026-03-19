from pathlib import Path

from memory_decay.cross_validator import run_kfold


def test_kfold_returns_fold_results():
    exp_dir = Path("experiments/exp_0338")
    result = run_kfold(exp_dir, k=3, cache_dir=Path("cache"))
    assert "fold_scores" in result
    assert len(result["fold_scores"]) == 3
    assert "mean" in result
    assert "std" in result
    assert 0 < result["mean"]["overall_score"] < 1
