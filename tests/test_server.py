"""Tests for the memory-decay FastAPI server."""

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


class TestHealthAndStats:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_stats_empty(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        assert r.json()["num_memories"] == 0


class TestStoreAndSearch:
    def test_store_memory(self, client):
        r = client.post("/store", json={
            "text": "I prefer dark mode",
            "importance": 0.8,
            "category": "preference",
            "mtype": "fact",
        })
        assert r.status_code == 200
        assert "id" in r.json()

    def test_search_finds_stored(self, client):
        client.post("/store", json={
            "text": "Python is great for data science",
            "importance": 0.7,
            "category": "fact",
            "mtype": "fact",
        })
        r = client.post("/search", json={
            "query": "Python programming",
            "top_k": 5,
        })
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) >= 1

    def test_search_empty(self, client):
        r = client.post("/search", json={"query": "nothing"})
        assert r.status_code == 200
        assert r.json()["results"] == []



class TestTick:
    def test_tick_advances(self, client):
        r = client.post("/tick", json={"count": 3})
        assert r.status_code == 200
        assert r.json()["current_tick"] == 3

    def test_auto_tick(self, client):
        r = client.post("/auto-tick")
        assert r.status_code == 200
        assert "ticks_applied" in r.json()


class TestForget:
    def test_forget_by_id(self, client):
        store_r = client.post("/store", json={
            "text": "secret info",
            "importance": 0.5,
            "category": "fact",
            "mtype": "fact",
        })
        memory_id = store_r.json()["id"]
        r = client.delete(f"/forget/{memory_id}")
        assert r.status_code == 200

    def test_forget_nonexistent(self, client):
        r = client.delete("/forget/nonexistent-id")
        assert r.status_code == 404


class TestStoreBatch:
    def test_batch_store(self, client):
        items = [
            {"text": "fact one", "importance": 0.8, "mtype": "fact"},
            {"text": "fact two", "importance": 0.6, "mtype": "fact"},
            {"text": "episode one", "importance": 0.7, "mtype": "episode"},
        ]
        r = client.post("/store-batch", json=items)
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 3
        assert len(data["ids"]) == 3

    def test_batch_store_searchable(self, client):
        items = [
            {"text": "batch search target", "importance": 0.9, "mtype": "fact"},
        ]
        client.post("/store-batch", json=items)
        r = client.post("/search", json={"query": "batch search target", "top_k": 5})
        assert r.status_code == 200
        assert len(r.json()["results"]) >= 1

    def test_batch_store_empty(self, client):
        r = client.post("/store-batch", json=[])
        assert r.status_code == 200
        assert r.json()["count"] == 0


class TestReset:
    def test_reset(self, client):
        client.post("/store", json={"text": "will be cleared", "mtype": "fact"})
        r = client.post("/reset")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        stats = client.get("/stats").json()
        assert stats["num_memories"] == 0
