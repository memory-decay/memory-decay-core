"""Tests for activation history tracking and admin API endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from memory_decay.decay import DecayEngine
from memory_decay.memory_store import MemoryStore
from memory_decay.server import create_app


dim = 8
embedder = lambda t: np.random.RandomState(hash(t) % 2**31).randn(dim).astype(np.float32)


def _emb(seed=42):
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# MemoryStore unit tests
# ---------------------------------------------------------------------------


class TestRecordActivationHistory:
    def test_records_snapshots(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        store.add_memory("m1", "hello", _emb(1), importance=0.7, mtype="episode", created_tick=0)
        store.add_memory("m2", "world", _emb(2), importance=0.5, mtype="fact", created_tick=0)

        count = store.record_activation_history(tick=0)
        assert count == 2

        history = store.get_activation_history("m1")
        assert len(history) == 1
        assert history[0]["tick"] == 0
        assert history[0]["retrieval_score"] == 1.0
        assert history[0]["storage_score"] == 1.0
        store.close()

    def test_skips_unchanged(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        store.add_memory("m1", "hello", _emb(1), importance=0.7, created_tick=0)

        store.record_activation_history(tick=0)
        # Record again without changing scores
        count = store.record_activation_history(tick=1)
        assert count == 0  # unchanged, should skip

        history = store.get_activation_history("m1")
        assert len(history) == 1  # only the first snapshot
        store.close()

    def test_records_after_change(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        store.add_memory("m1", "hello", _emb(1), importance=0.7, created_tick=0)

        store.record_activation_history(tick=0)
        store.set_retrieval_score("m1", 0.8)
        count = store.record_activation_history(tick=1)
        assert count == 1

        history = store.get_activation_history("m1")
        assert len(history) == 2
        assert history[1]["retrieval_score"] == 0.8
        store.close()

    def test_history_tick_range_filter(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        store.add_memory("m1", "hello", _emb(1), importance=0.7, created_tick=0)

        for tick in range(5):
            store.set_retrieval_score("m1", 1.0 - tick * 0.1)
            store.record_activation_history(tick=tick)

        # Filter by range
        history = store.get_activation_history("m1", start_tick=2, end_tick=4)
        assert len(history) == 3
        assert history[0]["tick"] == 2
        assert history[-1]["tick"] == 4
        store.close()

    def test_empty_store_returns_zero(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        count = store.record_activation_history(tick=0)
        assert count == 0
        store.close()


class TestGetAllMemories:
    def test_paginated_listing(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        for i in range(10):
            store.add_memory(f"m{i}", f"content {i}", _emb(i), created_tick=i)

        memories, total = store.get_all_memories(page=1, per_page=3)
        assert total == 10
        assert len(memories) == 3

        memories2, _ = store.get_all_memories(page=2, per_page=3)
        assert len(memories2) == 3
        # Pages should not overlap
        ids1 = {m["id"] for m in memories}
        ids2 = {m["id"] for m in memories2}
        assert ids1.isdisjoint(ids2)
        store.close()

    def test_filter_by_category(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        store.add_memory("m1", "a", _emb(1), category="pref", created_tick=0)
        store.add_memory("m2", "b", _emb(2), category="fact", created_tick=0)
        store.add_memory("m3", "c", _emb(3), category="pref", created_tick=0)

        memories, total = store.get_all_memories(category="pref")
        assert total == 2
        assert all(m["category"] == "pref" for m in memories)
        store.close()

    def test_filter_by_mtype(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        store.add_memory("m1", "a", _emb(1), mtype="fact", created_tick=0)
        store.add_memory("m2", "b", _emb(2), mtype="episode", created_tick=0)

        memories, total = store.get_all_memories(mtype="fact")
        assert total == 1
        assert memories[0]["mtype"] == "fact"
        store.close()


class TestGetMemorySummary:
    def test_summary_stats(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        store.add_memory("m1", "a", _emb(1), category="pref", importance=0.9, created_tick=0)
        store.add_memory("m2", "b", _emb(2), category="fact", importance=0.5, created_tick=0)
        store.set_retrieval_score("m2", 0.2)  # at-risk

        summary = store.get_memory_summary()
        assert summary["total_memories"] == 2
        assert summary["at_risk_count"] == 1
        assert len(summary["categories"]) == 2
        assert summary["avg_retrieval_score"] > 0
        store.close()

    def test_summary_empty_store(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        summary = store.get_memory_summary()
        assert summary["total_memories"] == 0
        assert summary["at_risk_count"] == 0
        assert summary["categories"] == []
        store.close()


# ---------------------------------------------------------------------------
# Integration: DecayEngine + activation history
# ---------------------------------------------------------------------------


class TestDecayWithHistory:
    def test_decay_records_history(self):
        store = MemoryStore(":memory:", embedding_dim=dim)
        store.add_memory("m1", "hello", _emb(1), importance=0.7, mtype="episode", created_tick=0)

        engine = DecayEngine(store=store, params={
            "lambda_fact": 0.05, "lambda_episode": 0.08,
            "alpha": 0.5, "stability_weight": 0.8,
            "stability_decay": 0.01, "stability_cap": 1.0,
        })

        # Record initial state
        store.record_activation_history(tick=0)

        for _ in range(5):
            engine.tick()
            store.record_activation_history(tick=engine.current_tick)

        history = store.get_activation_history("m1")
        assert len(history) == 6  # tick 0 + 5 decay ticks
        # Scores should be monotonically decreasing
        for i in range(1, len(history)):
            assert history[i]["retrieval_score"] <= history[i - 1]["retrieval_score"]
        store.close()


# ---------------------------------------------------------------------------
# Server admin endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    app = create_app(embedding_provider=None, _test_embedder=embedder)
    with TestClient(app) as c:
        yield c


def _store_memories(client, count=5):
    ids = []
    for i in range(count):
        r = client.post("/store", json={
            "text": f"memory content {i}",
            "importance": round(0.5 + (i % 5) * 0.1, 1),
            "mtype": "fact" if i % 2 == 0 else "episode",
            "category": "cat_a" if i < 3 else "cat_b",
        })
        ids.append(r.json()["id"])
    return ids


class TestAdminListMemories:
    def test_list_empty(self, client):
        r = client.get("/admin/memories")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["memories"] == []

    def test_list_with_pagination(self, client):
        _store_memories(client, 10)
        r = client.get("/admin/memories?page=1&per_page=3")
        data = r.json()
        assert data["total"] == 10
        assert len(data["memories"]) == 3
        assert data["page"] == 1

    def test_list_filter_category(self, client):
        _store_memories(client, 5)
        r = client.get("/admin/memories?category=cat_a")
        data = r.json()
        assert data["total"] == 3
        assert all(m["category"] == "cat_a" for m in data["memories"])

    def test_list_filter_mtype(self, client):
        _store_memories(client, 5)
        r = client.get("/admin/memories?mtype=fact")
        data = r.json()
        assert all(m["mtype"] == "fact" for m in data["memories"])


class TestAdminGetMemory:
    def test_get_single_memory(self, client):
        ids = _store_memories(client, 1)
        r = client.get(f"/admin/memories/{ids[0]}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == ids[0]
        assert "retrieval_score" in data
        assert "storage_score" in data

    def test_get_nonexistent(self, client):
        r = client.get("/admin/memories/nonexistent")
        assert r.status_code == 404


class TestAdminMemoryHistory:
    def test_history_after_ticks(self, client):
        ids = _store_memories(client, 1)
        # Run some ticks (which record history)
        client.post("/tick", json={"count": 5})

        r = client.get(f"/admin/memories/{ids[0]}/history")
        assert r.status_code == 200
        data = r.json()
        assert data["memory_id"] == ids[0]
        assert len(data["history"]) == 5  # history_interval=1, 5 ticks

    def test_history_tick_range(self, client):
        ids = _store_memories(client, 1)
        client.post("/tick", json={"count": 10})

        r = client.get(f"/admin/memories/{ids[0]}/history?start_tick=3&end_tick=7")
        data = r.json()
        ticks = [h["tick"] for h in data["history"]]
        assert all(3 <= t <= 7 for t in ticks)

    def test_history_nonexistent_memory(self, client):
        r = client.get("/admin/memories/nonexistent/history")
        assert r.status_code == 404


class TestAdminSummary:
    def test_summary(self, client):
        _store_memories(client, 5)
        r = client.get("/admin/history/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["total_memories"] == 5
        assert "avg_retrieval_score" in data
        assert "at_risk_count" in data
        assert "categories" in data
        assert "current_tick" in data

    def test_summary_empty(self, client):
        r = client.get("/admin/history/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["total_memories"] == 0


class TestExistingEndpointsUnchanged:
    """Verify backward compatibility — existing endpoints still work."""

    def test_health(self, client):
        assert client.get("/health").status_code == 200

    def test_stats(self, client):
        assert client.get("/stats").status_code == 200

    def test_store_and_search(self, client):
        client.post("/store", json={"text": "hello", "mtype": "fact"})
        r = client.post("/search", json={"query": "hello"})
        assert r.status_code == 200

    def test_tick(self, client):
        r = client.post("/tick", json={"count": 1})
        assert r.json()["current_tick"] == 1

    def test_reset(self, client):
        r = client.post("/reset")
        assert r.json()["status"] == "ok"
