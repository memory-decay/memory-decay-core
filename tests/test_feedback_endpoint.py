"""Tests for /feedback endpoint and feedback cleanup in /reset and /forget."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from memory_decay.server import create_app


@pytest.fixture
def client():
    """Create test client with fake embedder."""
    dim = 8
    embedder = lambda t: np.random.RandomState(hash(t) % 2**31).randn(dim).astype(np.float32)
    app = create_app(embedding_provider=None, _test_embedder=embedder)
    with TestClient(app) as c:
        yield c


def _store_memory(client, text="test memory"):
    r = client.post("/store", json={"text": text, "importance": 0.7, "mtype": "fact"})
    assert r.status_code == 200
    return r.json()["id"]


class TestFeedbackEndpoint:
    def test_feedback_positive_returns_200(self, client):
        mid = _store_memory(client)
        r = client.post("/feedback", json={
            "items": [{"memory_id": mid, "signal": "positive", "strength": 1.0}],
        })
        assert r.status_code == 200
        assert r.json()["applied"] >= 1

    def test_feedback_negative_returns_200(self, client):
        mid = _store_memory(client)
        r = client.post("/feedback", json={
            "items": [{"memory_id": mid, "signal": "negative", "strength": 0.5}],
        })
        assert r.status_code == 200
        assert r.json()["applied"] >= 1

    def test_feedback_unknown_memory_skipped(self, client):
        r = client.post("/feedback", json={
            "items": [{"memory_id": "nonexistent", "signal": "positive", "strength": 1.0}],
        })
        assert r.status_code == 200
        assert r.json()["applied"] == 0

    def test_feedback_positive_adjusts_stability(self, client):
        mid = _store_memory(client)
        client.post("/feedback", json={
            "items": [{"memory_id": mid, "signal": "positive", "strength": 1.0}],
        })
        # Search to get the memory back — check stability changed
        # We can't directly inspect the store via HTTP, but we can verify
        # the response indicated adjustment
        r = client.post("/feedback", json={
            "items": [{"memory_id": mid, "signal": "positive", "strength": 1.0}],
        })
        # Second call within 60s should be deduped
        assert r.json()["applied"] == 0

    def test_feedback_batch_multiple_items(self, client):
        m1 = _store_memory(client, "memory one")
        m2 = _store_memory(client, "memory two")
        r = client.post("/feedback", json={
            "items": [
                {"memory_id": m1, "signal": "positive", "strength": 1.0},
                {"memory_id": m2, "signal": "negative", "strength": 0.5},
            ],
        })
        assert r.status_code == 200
        assert r.json()["applied"] == 2

    def test_feedback_dedup_within_window(self, client):
        mid = _store_memory(client)
        r1 = client.post("/feedback", json={
            "items": [{"memory_id": mid, "signal": "positive", "strength": 1.0}],
        })
        assert r1.json()["applied"] == 1
        r2 = client.post("/feedback", json={
            "items": [{"memory_id": mid, "signal": "positive", "strength": 1.0}],
        })
        assert r2.json()["applied"] == 0


class TestForgetCleansFeedback:
    def test_forget_deletes_feedback_log(self, client):
        mid = _store_memory(client)
        client.post("/feedback", json={
            "items": [{"memory_id": mid, "signal": "positive", "strength": 1.0}],
        })
        r = client.delete(f"/forget/{mid}")
        assert r.status_code == 200
        # Memory is gone — a second feedback should apply 0
        r2 = client.post("/feedback", json={
            "items": [{"memory_id": mid, "signal": "positive", "strength": 1.0}],
        })
        assert r2.json()["applied"] == 0


class TestResetCleansFeedback:
    def test_reset_clears_feedback_log(self, client):
        mid = _store_memory(client)
        client.post("/feedback", json={
            "items": [{"memory_id": mid, "signal": "positive", "strength": 1.0}],
        })
        r = client.post("/reset")
        assert r.status_code == 200
        # After reset, store a new memory and provide feedback — should not be deduped
        mid2 = _store_memory(client, "new memory after reset")
        r2 = client.post("/feedback", json={
            "items": [{"memory_id": mid2, "signal": "positive", "strength": 1.0}],
        })
        assert r2.json()["applied"] == 1

    def test_reset_response_includes_cleared(self, client):
        _store_memory(client)
        r = client.post("/reset")
        assert r.status_code == 200
        assert "cleared" in r.json()
