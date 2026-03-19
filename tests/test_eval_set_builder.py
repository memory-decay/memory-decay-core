"""Tests for EvalSetBuilder."""

from __future__ import annotations

from pathlib import Path


def test_identify_memory_weakness_returns_structure():
    """EvalSetBuilder.identify_memory_weakness returns required keys."""
    from memory_decay.eval_set_builder import EvalSetBuilder
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.jsonl"
        history_path.write_text("")
        builder = EvalSetBuilder(history_path=history_path)
        result = builder.identify_memory_weakness()
        assert "type" in result
        assert "description" in result
        assert result["type"] in ["recall", "mrr", "correlation", "selectivity", "precision_lift"]


def test_identify_memory_weakness_from_real_history():
    """With real history data, returns the bottleneck metric."""
    from memory_decay.eval_set_builder import EvalSetBuilder

    builder = EvalSetBuilder(history_path=Path("experiments/history.jsonl"))
    result = builder.identify_memory_weakness()
    assert result["type"] in ["recall", "mrr", "correlation", "selectivity", "precision_lift"]
    assert "gap" in result
    assert result["gap"] >= 0


def test_find_unexplored_structural_slots():
    """Identifies decay function types not yet systematically explored."""
    from memory_decay.eval_set_builder import EvalSetBuilder

    builder = EvalSetBuilder(history_path=Path("experiments/history.jsonl"))
    unexplored = builder._find_unexplored_structural_slots()
    assert isinstance(unexplored, list)
    assert all(isinstance(x, str) for x in unexplored)
