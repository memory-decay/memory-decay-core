"""Tests for the human calibration benchmark pipeline."""

from pathlib import Path

from memory_decay.calibration_pipeline import (
    run_calibration_benchmark,
    run_calibration_benchmark_suite,
)


def test_run_calibration_benchmark_writes_summary(tmp_path, monkeypatch):
    calls = []

    def fake_run_human_calibration(events_path, output_dir, *, iterations, seed):
        calls.append(("human", events_path, output_dir, iterations, seed))
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        best_params = out / "best_params.json"
        best_params.write_text('{"lambda_fact": 0.01}', encoding="utf-8")
        return {
            "best_params": {"lambda_fact": 0.01},
            "valid_metrics": {"nll": 0.5},
            "test_metrics": {"nll": 0.4, "brier": 0.2, "ece": 0.1},
        }

    def fake_run_experiment(**kwargs):
        calls.append(("experiment", kwargs))
        calibrated = kwargs.get("calibrated_params_path") is not None
        score = 0.39 if calibrated else 0.10
        return {
            "initial_overall_score": score,
            "initial_score_summary": {
                "overall_score": score,
                "retrieval_score": 0.4 if calibrated else 0.1,
                "plausibility_score": 0.3 if calibrated else 0.12,
            },
        }

    monkeypatch.setattr(
        "memory_decay.calibration_pipeline.run_human_calibration",
        fake_run_human_calibration,
    )
    monkeypatch.setattr(
        "memory_decay.calibration_pipeline.run_experiment",
        fake_run_experiment,
    )

    result = run_calibration_benchmark(
        human_events_path="data/human_reviews_smoke.jsonl",
        synthetic_dataset_path="data/memories_50.jsonl",
        output_dir=str(tmp_path / "outputs"),
        iterations=3,
        seed=42,
    )

    summary_path = tmp_path / "outputs" / "comparison_summary.json"
    assert summary_path.exists()
    assert result["delta"]["overall_score"] == 0.29
    assert result["baseline"]["overall_score"] == 0.10
    assert result["calibrated"]["overall_score"] == 0.39
    assert any(call[0] == "human" for call in calls)
    assert sum(1 for call in calls if call[0] == "experiment") == 2


def test_run_calibration_benchmark_uses_best_params_artifact(tmp_path, monkeypatch):
    seen = {}

    def fake_run_human_calibration(events_path, output_dir, *, iterations, seed):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        best_params = out / "best_params.json"
        best_params.write_text('{"lambda_fact": 0.02}', encoding="utf-8")
        return {
            "best_params": {"lambda_fact": 0.02},
            "valid_metrics": {"nll": 0.5},
            "test_metrics": {"nll": 0.4, "brier": 0.2, "ece": 0.1},
        }

    def fake_run_experiment(**kwargs):
        if kwargs.get("calibrated_params_path"):
            seen["path"] = kwargs["calibrated_params_path"]
        return {
            "initial_overall_score": 0.2,
            "initial_score_summary": {
                "overall_score": 0.2,
                "retrieval_score": 0.2,
                "plausibility_score": 0.2,
            },
        }

    monkeypatch.setattr(
        "memory_decay.calibration_pipeline.run_human_calibration",
        fake_run_human_calibration,
    )
    monkeypatch.setattr(
        "memory_decay.calibration_pipeline.run_experiment",
        fake_run_experiment,
    )

    run_calibration_benchmark(
        human_events_path="data/human_reviews_smoke.jsonl",
        synthetic_dataset_path="data/memories_50.jsonl",
        output_dir=str(tmp_path / "outputs"),
    )

    assert seen["path"].endswith("best_params.json")


def test_run_calibration_benchmark_suite_writes_per_dataset_outputs(tmp_path, monkeypatch):
    calls = []

    def fake_run_calibration_benchmark(**kwargs):
        calls.append(kwargs["synthetic_dataset_path"])
        return {
            "baseline": {"overall_score": 0.1},
            "calibrated": {"overall_score": 0.3},
            "delta": {"overall_score": 0.2},
        }

    monkeypatch.setattr(
        "memory_decay.calibration_pipeline.run_calibration_benchmark",
        fake_run_calibration_benchmark,
    )

    result = run_calibration_benchmark_suite(
        human_events_path="data/human_reviews_smoke.jsonl",
        synthetic_dataset_paths=["data/memories_50.jsonl", "data/memories_500.jsonl"],
        output_dir=str(tmp_path / "suite"),
        iterations=3,
        seed=42,
    )

    summary_path = tmp_path / "suite" / "suite_summary.json"
    assert summary_path.exists()
    assert set(result["runs"]) == {"memories_50", "memories_500"}
    assert calls == ["data/memories_50.jsonl", "data/memories_500.jsonl"]
