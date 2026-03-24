"""Test DecayEngine with SQLite-backed MemoryStore."""
import numpy as np
import pytest
from memory_decay.memory_store import MemoryStore
from memory_decay.decay import DecayEngine


def _emb(seed=42, dim=384):
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)  # pre-normalize


def test_decay_with_store():
    store = MemoryStore(":memory:", embedding_dim=384)
    store.add_memory("m1", "hello", _emb(1), importance=0.7, mtype="episode", created_tick=0)
    store.add_memory("m2", "world", _emb(2), importance=0.3, mtype="fact", created_tick=0)

    engine = DecayEngine(store=store, params={
        "lambda_fact": 0.05, "lambda_episode": 0.08,
        "alpha": 0.5, "stability_weight": 0.8,
        "stability_decay": 0.01, "stability_cap": 1.0,
    })

    for _ in range(10):
        engine.tick()

    node = store.get_node("m1")
    assert node["retrieval_score"] < 1.0
    assert node["retrieval_score"] > 0.0
    assert node["stability_score"] >= 0.0  # stability decayed but >= 0
    store.close()


def test_decay_preserves_untouched_memories():
    store = MemoryStore(":memory:", embedding_dim=384)
    store.add_memory("m1", "early", _emb(1), created_tick=0)
    store.add_memory("m2", "future", _emb(2), created_tick=100)  # not yet active

    engine = DecayEngine(store=store, params={
        "lambda_fact": 0.05, "lambda_episode": 0.08,
        "alpha": 0.5, "stability_weight": 0.8,
        "stability_decay": 0.01, "stability_cap": 1.0,
    })

    for _ in range(5):
        engine.tick()

    m1 = store.get_node("m1")
    m2 = store.get_node("m2")
    assert m1["retrieval_score"] < 1.0  # decayed
    assert m2["retrieval_score"] == 1.0  # untouched (created_tick > current_tick)
    store.close()
