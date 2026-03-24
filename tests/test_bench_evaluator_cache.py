"""Test that evaluate_cached skips re-evaluation when all results are cached."""

from unittest.mock import patch
import json
from memory_decay.bench_evaluator import evaluate_cached, BENCHMARKS


def test_evaluate_cached_skips_when_all_cached(tmp_path):
    """When all benchmark results exist in cache files, evaluate() should NOT be called."""
    exp_dir = tmp_path / "exp_test"
    exp_dir.mkdir()
    run_prefix = "run_0"

    for bench in BENCHMARKS:
        result_file = exp_dir / f"{run_prefix}_{bench}_result.json"
        result_file.write_text(json.dumps({
            "summary": {"accuracy": 85.0, "totalQuestions": 100, "correctCount": 85},
            "retrieval": {"mrr": 0.75},
        }))

    with patch("memory_decay.bench_evaluator.evaluate") as mock_eval:
        result = evaluate_cached(str(exp_dir), run_prefix=run_prefix)
        mock_eval.assert_not_called()

    assert result.bench_score > 0
