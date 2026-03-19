from pathlib import Path

from memory_decay.cross_validator import run_kfold


def test_kfold_returns_fold_results(monkeypatch):
    dataset = [
        {"id": "f1", "type": "fact", "recall_query": "q1"},
        {"id": "f2", "type": "fact", "recall_query": "q2"},
        {"id": "f3", "type": "fact", "recall_query": "q3"},
        {"id": "e1", "type": "episode", "recall_query": "q4"},
        {"id": "e2", "type": "episode", "recall_query": "q5"},
        {"id": "e3", "type": "episode", "recall_query": "q6"},
    ]

    def fake_load_raw_dataset(path):
        return dataset

    def fake_run_experiment_with_split(*args, **kwargs):
        test_size = len(kwargs["test_items"] if "test_items" in kwargs else args[2])
        return {
            "overall_score": 0.20 + test_size * 0.01,
            "retrieval_score": 0.15,
            "plausibility_score": 0.55,
            "recall_mean": 0.30,
            "mrr_mean": 0.22,
            "corr_score": 0.12,
            "retention_auc": 0.25,
            "selectivity_score": 0.18,
            "robustness_score": 0.0,
            "eval_v2_score": 0.18,
        }

    monkeypatch.setattr("memory_decay.cache_builder.load_raw_dataset", fake_load_raw_dataset)
    monkeypatch.setattr("memory_decay.runner.run_experiment_with_split", fake_run_experiment_with_split)

    exp_dir = Path("experiments/exp_0338")
    result = run_kfold(exp_dir, k=3, cache_dir=Path("cache"))
    assert "fold_scores" in result
    assert len(result["fold_scores"]) == 3
    assert "mean" in result
    assert "std" in result
    assert 0 < result["mean"]["overall_score"] < 1
    assert "eval_v2_score" in result["mean"]
    assert "retention_auc" in result["mean"]
    assert "selectivity_score" in result["mean"]
    assert "worst_fold" in result
    assert "fold_deltas" in result
