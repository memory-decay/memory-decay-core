"""Tests for SQLite-backed MemoryStore."""
import numpy as np
import pytest
from memory_decay.memory_store import MemoryStore


def _random_embedding(dim=384, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)


class TestMemoryStoreBasic:
    def test_create_in_memory(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        assert store.num_memories == 0
        store.close()

    def test_add_and_count(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "hello world", _random_embedding(384, 1),
                         user_id="u1", mtype="fact", importance=0.8)
        assert store.num_memories == 1
        store.close()

    def test_search_returns_results(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "I love baking cakes", _random_embedding(384, 1),
                         user_id="u1")
        store.add_memory("m2", "The weather is sunny", _random_embedding(384, 2),
                         user_id="u1")
        results = store.search(_random_embedding(384, 1), top_k=2)
        assert len(results) >= 1
        assert results[0]["id"] == "m1"  # closest to itself
        store.close()

    def test_search_self_similarity_is_one(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        emb = _random_embedding(384, 1)
        store.add_memory("m1", "test", emb, user_id="u1")
        results = store.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(1.0, abs=0.001)
        store.close()

    def test_clear(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "test", _random_embedding(384, 1), user_id="u1")
        cleared = store.clear()
        assert cleared == 1
        assert store.num_memories == 0
        store.close()

    def test_clear_by_user(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "a", _random_embedding(384, 1), user_id="u1")
        store.add_memory("m2", "b", _random_embedding(384, 2), user_id="u2")
        store.clear(user_id="u1")
        assert store.num_memories == 1
        store.close()

    def test_get_node(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "hello", _random_embedding(384, 1),
                         user_id="u1", mtype="fact", importance=0.9,
                         speaker="Alice", created_tick=5)
        node = store.get_node("m1")
        assert node is not None
        assert node["content"] == "hello"
        assert node["mtype"] == "fact"
        assert node["importance"] == pytest.approx(0.9)
        assert node["speaker"] == "Alice"
        assert node["created_tick"] == 5
        store.close()

    def test_get_node_missing(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        assert store.get_node("nonexistent") is None
        store.close()

    def test_search_with_activation_weight(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        emb = _random_embedding(384, 1)
        store.add_memory("m1", "high", emb, user_id="u1")
        store.add_memory("m2", "low", emb, user_id="u1")  # same embedding
        # Set m1 high retrieval, m2 low
        store.set_retrieval_score("m1", 0.9)
        store.set_retrieval_score("m2", 0.1)
        results = store.search(emb, top_k=2, activation_weight=1.0)
        assert results[0]["id"] == "m1"  # higher activation wins
        store.close()

    def test_search_with_user_id_filter(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        emb = _random_embedding(384, 1)
        store.add_memory("m1", "user1 mem", emb, user_id="u1")
        store.add_memory("m2", "user2 mem", _random_embedding(384, 2), user_id="u2")
        results = store.search(emb, top_k=10, user_id="u1")
        assert all(r["id"] != "m2" for r in results)
        store.close()

    def test_search_with_current_tick_filter(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "past", _random_embedding(384, 1),
                         user_id="u1", created_tick=5)
        store.add_memory("m2", "future", _random_embedding(384, 2),
                         user_id="u1", created_tick=100)
        results = store.search(_random_embedding(384, 1), top_k=10, current_tick=10)
        ids = [r["id"] for r in results]
        assert "m1" in ids
        assert "m2" not in ids
        store.close()


class TestMemoryStoreScores:
    def test_set_retrieval_score(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "test", _random_embedding(384, 1))
        store.set_retrieval_score("m1", 0.42)
        node = store.get_node("m1")
        assert node["retrieval_score"] == pytest.approx(0.42)
        store.close()

    def test_set_storage_score(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "test", _random_embedding(384, 1))
        store.set_storage_score("m1", 0.55)
        node = store.get_node("m1")
        assert node["storage_score"] == pytest.approx(0.55)
        store.close()

    def test_set_activation(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "test", _random_embedding(384, 1))
        store.set_activation("m1", 0.33)
        node = store.get_node("m1")
        assert node["retrieval_score"] == pytest.approx(0.33)
        store.close()

    def test_batch_update_scores(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "a", _random_embedding(384, 1))
        store.add_memory("m2", "b", _random_embedding(384, 2))
        updates = [
            (0.5, 0.6, 0.1, "m1"),
            (0.3, 0.4, 0.2, "m2"),
        ]
        store.batch_update_scores(updates)
        n1 = store.get_node("m1")
        n2 = store.get_node("m2")
        assert n1["retrieval_score"] == pytest.approx(0.5)
        assert n1["storage_score"] == pytest.approx(0.6)
        assert n1["stability_score"] == pytest.approx(0.1)
        assert n2["retrieval_score"] == pytest.approx(0.3)
        store.close()

    def test_get_all_for_decay(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "a", _random_embedding(384, 1), created_tick=0)
        store.add_memory("m2", "b", _random_embedding(384, 2), created_tick=5)
        store.add_memory("m3", "c", _random_embedding(384, 3), created_tick=100)
        rows = store.get_all_for_decay(current_tick=10)
        ids = {r["id"] for r in rows}
        assert "m1" in ids
        assert "m2" in ids
        assert "m3" not in ids
        store.close()


class TestMemoryStoreMetadata:
    def test_set_and_get_metadata(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.set_metadata("version", "1.0")
        assert store.get_metadata("version") == "1.0"
        store.close()

    def test_get_metadata_default(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        assert store.get_metadata("missing", "fallback") == "fallback"
        store.close()

    def test_metadata_overwrite(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.set_metadata("k", "v1")
        store.set_metadata("k", "v2")
        assert store.get_metadata("k") == "v2"
        store.close()


class TestMemoryStoreAssociations:
    def test_add_with_associations(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "first", _random_embedding(384, 1))
        store.add_memory("m2", "second", _random_embedding(384, 2),
                         associations=[("m1", 0.8)])
        # Check both directions exist
        row = store.db.execute(
            "SELECT weight FROM associations WHERE source_id='m2' AND target_id='m1'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(0.8)
        rev = store.db.execute(
            "SELECT weight FROM associations WHERE source_id='m1' AND target_id='m2'"
        ).fetchone()
        assert rev is not None
        store.close()
