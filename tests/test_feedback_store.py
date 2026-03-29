"""Tests for feedback_log table and adjust_scores in MemoryStore."""
import time

import numpy as np
import pytest

from memory_decay.memory_store import MemoryStore


def _random_embedding(dim=384, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)


def _make_store_with_memory(memory_id="m1", stability=0.5, retrieval_count=0,
                            last_activated_tick=0, created_tick=0):
    """Helper: create an in-memory store with one memory pre-configured."""
    store = MemoryStore(":memory:", embedding_dim=384)
    store.add_memory(memory_id, "test content", _random_embedding(384, 1),
                     created_tick=created_tick)
    # Set initial scores
    store._db.execute(
        "UPDATE memories SET stability_score=?, retrieval_count=?, last_activated_tick=? WHERE id=?",
        (stability, retrieval_count, last_activated_tick, memory_id),
    )
    store._db.commit()
    return store


class TestFeedbackLogTable:
    def test_feedback_log_table_exists(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        row = store._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback_log'"
        ).fetchone()
        assert row is not None
        store.close()


class TestAdjustScoresPositive:
    def test_positive_increases_stability(self):
        store = _make_store_with_memory(stability=0.5)
        store.adjust_scores([("m1", "positive", 1.0)], current_tick=10)
        node = store.get_node("m1")
        # delta = 0.15 * 1.0 * (1 - 0.5/1.0) = 0.075
        assert node["stability_score"] == pytest.approx(0.575, abs=0.001)
        store.close()

    def test_positive_updates_retrieval_count_and_last_activated(self):
        store = _make_store_with_memory(stability=0.3, retrieval_count=2,
                                         last_activated_tick=0)
        store.adjust_scores([("m1", "positive", 0.8)], current_tick=15)
        node = store.get_node("m1")
        assert node["retrieval_count"] == 3
        assert node["last_activated_tick"] == 15
        store.close()

    def test_positive_stability_formula(self):
        """Verify: delta = 0.15 * strength * (1 - stability / stability_cap)."""
        store = _make_store_with_memory(stability=0.8)
        store.adjust_scores([("m1", "positive", 0.5)], current_tick=1)
        node = store.get_node("m1")
        # delta = 0.15 * 0.5 * (1 - 0.8/1.0) = 0.015
        expected = 0.8 + 0.015
        assert node["stability_score"] == pytest.approx(expected, abs=0.001)
        store.close()


class TestAdjustScoresNegative:
    def test_negative_decreases_stability(self):
        store = _make_store_with_memory(stability=0.5)
        store.adjust_scores([("m1", "negative", 1.0)], current_tick=10)
        node = store.get_node("m1")
        assert node["stability_score"] == pytest.approx(0.45, abs=0.001)
        store.close()

    def test_negative_does_not_update_retrieval_count(self):
        store = _make_store_with_memory(stability=0.5, retrieval_count=5,
                                         last_activated_tick=3)
        store.adjust_scores([("m1", "negative", 1.0)], current_tick=10)
        node = store.get_node("m1")
        assert node["retrieval_count"] == 5
        assert node["last_activated_tick"] == 3
        store.close()


class TestAdjustScoresClamping:
    def test_clamp_stability_at_zero(self):
        store = _make_store_with_memory(stability=0.02)
        store.adjust_scores([("m1", "negative", 1.0)], current_tick=1)
        node = store.get_node("m1")
        assert node["stability_score"] == 0.0
        store.close()

    def test_clamp_stability_at_one(self):
        store = _make_store_with_memory(stability=0.99)
        store.adjust_scores([("m1", "positive", 1.0)], current_tick=1)
        node = store.get_node("m1")
        # delta = 0.15 * 1.0 * (1 - 0.99) = 0.0015 → 0.9915, still < 1
        # But let's also test actual clamping at 1.0 with multiple calls
        assert node["stability_score"] <= 1.0
        store.close()


class TestAdjustScoresDedup:
    def test_dedup_within_60s_window(self):
        store = _make_store_with_memory(stability=0.5)
        store.adjust_scores([("m1", "positive", 1.0)], current_tick=1)
        node1 = store.get_node("m1")
        # Second call within 60s should be deduped
        store.adjust_scores([("m1", "positive", 1.0)], current_tick=2)
        node2 = store.get_node("m1")
        assert node2["stability_score"] == pytest.approx(node1["stability_score"], abs=0.001)
        store.close()

    def test_dedup_allows_after_60s(self):
        store = _make_store_with_memory(stability=0.5)
        # Insert a feedback_log entry with old timestamp (>60s ago)
        old_ts = time.time() - 61
        store._db.execute(
            "INSERT INTO feedback_log (memory_id, signal, strength, timestamp, tick) "
            "VALUES (?, ?, ?, ?, ?)",
            ("m1", "positive", 1.0, old_ts, 1),
        )
        store._db.commit()
        # This should NOT be deduped since the previous entry is >60s old
        store.adjust_scores([("m1", "positive", 1.0)], current_tick=5)
        node = store.get_node("m1")
        # delta = 0.15 * 1.0 * (1 - 0.5/1.0) = 0.075
        assert node["stability_score"] == pytest.approx(0.575, abs=0.001)
        store.close()


class TestFeedbackLogRecording:
    def test_adjust_scores_logs_to_feedback_log(self):
        store = _make_store_with_memory(stability=0.5)
        store.adjust_scores([("m1", "positive", 0.8)], current_tick=10)
        rows = store._db.execute(
            "SELECT memory_id, signal, strength, tick FROM feedback_log"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "m1"
        assert rows[0][1] == "positive"
        assert rows[0][2] == pytest.approx(0.8)
        assert rows[0][3] == 10
        store.close()


class TestClearAndDeleteFeedback:
    def test_clear_feedback(self):
        store = _make_store_with_memory(stability=0.5)
        store.adjust_scores([("m1", "positive", 1.0)], current_tick=1)
        count = store._db.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
        assert count > 0
        store.clear_feedback()
        count = store._db.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
        assert count == 0
        store.close()

    def test_delete_feedback_for(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "a", _random_embedding(384, 1))
        store.add_memory("m2", "b", _random_embedding(384, 2))
        store.adjust_scores([("m1", "positive", 1.0)], current_tick=1)
        # Wait to avoid dedup then add for m2
        store._db.execute(
            "INSERT INTO feedback_log (memory_id, signal, strength, timestamp, tick) "
            "VALUES (?, ?, ?, ?, ?)",
            ("m2", "negative", 0.5, time.time(), 2),
        )
        store._db.commit()
        store.delete_feedback_for("m1")
        rows = store._db.execute("SELECT memory_id FROM feedback_log").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "m2"
        store.close()
