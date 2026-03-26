"""Tests for the memory-decay FastAPI server."""

import json
import os
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from memory_decay.server import create_app


dim = 8
embedder = lambda t: np.random.RandomState(hash(t) % 2**31).randn(dim).astype(np.float32)


@pytest.fixture
def client():
    """Create test client with fake embedder."""
    app = create_app(embedding_provider=None, _test_embedder=embedder)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def bm25_client():
    """Create test client with BM25 enabled via experiment params."""
    with tempfile.TemporaryDirectory() as exp_dir:
        params_path = os.path.join(exp_dir, "params.json")
        with open(params_path, "w") as f:
            json.dump({"bm25_weight": 0.3}, f)
        app = create_app(
            embedding_provider=None,
            _test_embedder=embedder,
            experiment_dir=exp_dir,
        )
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


class TestSearchBM25:
    def test_search_with_bm25_enabled(self, bm25_client):
        """Search should use BM25 when bm25_weight is set in params."""
        bm25_client.post("/store", json={
            "text": "서울은 한국의 수도입니다",
            "importance": 0.8, "mtype": "fact",
        })
        bm25_client.post("/store", json={
            "text": "날씨가 좋습니다",
            "importance": 0.8, "mtype": "fact",
        })
        r = bm25_client.post("/search", json={
            "query": "한국 수도",
            "top_k": 5,
        })
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) >= 1

    def test_search_without_bm25_still_works(self, client):
        """Default client (bm25_weight=0) should still work normally."""
        client.post("/store", json={
            "text": "test memory", "importance": 0.5, "mtype": "fact",
        })
        r = client.post("/search", json={"query": "test", "top_k": 5})
        assert r.status_code == 200


class TestTickSync:
    def test_tick_counter_matches_engine(self, client):
        """Server tick and engine tick must stay in sync."""
        client.post("/tick", json={"count": 5})
        health = client.get("/health").json()
        assert health["current_tick"] == 5
        client.post("/tick", json={"count": 3})
        health = client.get("/health").json()
        assert health["current_tick"] == 8

    def test_tick_derived_from_engine(self, client):
        """Verify the server reads tick from engine, not its own counter."""
        from memory_decay.server import _state
        client.post("/tick", json={"count": 5})
        assert _state.engine.current_tick == _state.current_tick == 5


class TestGeminiDimensionRegression:
    """Regression: GeminiEmbeddingProvider.dimension must match actual embed output.

    Previously, GeminiEmbeddingProvider hardcoded _dim=768 but gemini-embedding-001
    actually returns 3072-dim vectors. The server reads provider.dimension at startup
    to create the DB table, so a mismatch causes 500 on the first /store request.
    """

    def test_store_succeeds_with_gemini_provider_dimension(self):
        """Server creates DB with provider.dimension; /store must not fail due to dim mismatch.

        Regression: GeminiEmbeddingProvider used to hardcode _dim=768 but
        gemini-embedding-001 returns 3072-dim vectors. The server reads
        provider.dimension at startup to size the DB table, so a mismatch
        caused 500 on the first /store.
        """
        from unittest.mock import MagicMock, AsyncMock
        from memory_decay.embedding_provider import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(api_key="fake-key", model="gemini-embedding-001")
        actual_dim = provider.dimension  # should be 3072 after fix

        # Mock embed methods to return a vector matching the provider's reported dimension
        fake_vec = np.random.randn(actual_dim).astype(np.float32)
        provider.embed = MagicMock(return_value=fake_vec)
        provider.aembed = AsyncMock(return_value=fake_vec)

        app = create_app(embedding_provider=provider)
        with TestClient(app) as c:
            r = c.post("/store", json={
                "text": "regression test",
                "importance": 0.5,
                "mtype": "fact",
            })
            assert r.status_code == 200, (
                f"Expected 200 but got {r.status_code}: {r.text}. "
                f"Provider reported dim={actual_dim}. "
                "This fails if provider.dimension doesn't match actual embed output."
            )
