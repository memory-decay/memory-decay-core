"""Tests for graph persistence (pickle-based save/load)."""

import json

import numpy as np
import pytest

from memory_decay.graph import MemoryGraph
from memory_decay.decay import DecayEngine
from memory_decay.persistence import MemoryPersistence


def _make_graph_with_memories(embedder) -> tuple[MemoryGraph, DecayEngine]:
    g = MemoryGraph(embedder=embedder)
    g.add_memory("m1", "fact", "Python is a programming language", 0.8, 0)
    g.add_memory("m2", "episode", "Had coffee this morning", 0.3, 1)
    e = DecayEngine(g)
    return g, e


def _fake_embedder(dim=8):
    return lambda t: np.random.RandomState(hash(t) % 2**31).randn(dim).astype(np.float32)


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        embedder = _fake_embedder()
        g1, e1 = _make_graph_with_memories(embedder)
        e1.tick()
        e1.tick()

        p = MemoryPersistence(tmp_path)
        p.save(g1, current_tick=2)

        g2 = MemoryGraph(embedder=embedder)
        meta = p.load(g2)

        assert g2.num_memories == g1.num_memories
        assert meta["current_tick"] == 2
        assert "saved_at" in meta

    def test_load_nonexistent_returns_none(self, tmp_path):
        g = MemoryGraph(embedder=lambda t: np.zeros(8, dtype=np.float32))
        p = MemoryPersistence(tmp_path)
        meta = p.load(g)
        assert meta is None

    def test_meta_json_written(self, tmp_path):
        embedder = lambda t: np.zeros(8, dtype=np.float32)
        g, _ = _make_graph_with_memories(embedder)

        p = MemoryPersistence(tmp_path)
        p.save(g, current_tick=5)

        meta_path = tmp_path / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["current_tick"] == 5
        assert meta["num_memories"] == 2

    def test_save_creates_directory(self, tmp_path):
        target = tmp_path / "nested" / "dir"
        embedder = lambda t: np.zeros(8, dtype=np.float32)
        g = MemoryGraph(embedder=embedder)
        g.add_memory("m1", "fact", "hello", 0.5, 0)

        p = MemoryPersistence(target)
        p.save(g, current_tick=0)
        assert (target / "graph.pkl").exists()
