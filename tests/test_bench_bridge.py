"""Tests for bench_bridge: CLI bridge for MemoryBench → MemoryStore."""

import json
import os
import tempfile

import numpy as np
import pytest

from memory_decay.bench_bridge import (
    CachedEmbeddingProvider,
    prepare,
    search_memories,
)
from memory_decay.embedding_provider import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic fake embedding provider for tests."""

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._call_count = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        self._call_count += 1
        rng = np.random.RandomState(hash(text) % (2**31))
        vec = rng.randn(self._dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def fake_provider():
    raw = FakeEmbeddingProvider(dim=64)
    return CachedEmbeddingProvider(provider=raw, model_name="fake-64")


@pytest.fixture
def fake_provider_with_cache(tmp_path):
    raw = FakeEmbeddingProvider(dim=64)
    cache_db = str(tmp_path / "cache.db")
    return CachedEmbeddingProvider(
        provider=raw, cache_db_path=cache_db, model_name="fake-64"
    )


def _make_messages():
    return [
        {
            "text": "[User] I love baking cakes",
            "importance": 0.7,
            "mtype": "episode",
            "created_tick": 0,
            "speaker": "User",
        },
        {
            "text": "[Assistant] That's great! What kind of cakes?",
            "importance": 0.4,
            "mtype": "episode",
            "created_tick": 0,
            "speaker": "Assistant",
        },
        {
            "text": "[User] Chocolate cakes are my favorite",
            "importance": 0.8,
            "mtype": "episode",
            "created_tick": 1,
            "speaker": "User",
        },
    ]


class TestPrepare:
    def test_basic_prepare(self, tmp_path, fake_provider):
        db_path = str(tmp_path / "test.db")
        messages = _make_messages()
        params = {"lambda_fact": 0.05, "lambda_episode": 0.2}

        result = prepare(
            db_path=db_path,
            messages=messages,
            params=params,
            embedding_provider=fake_provider,
            simulate_ticks=0,
        )

        assert result["status"] == "ok"
        assert result["memories_count"] == 3
        assert result["cached"] is False

    def test_prepare_with_ticks(self, tmp_path, fake_provider):
        db_path = str(tmp_path / "test.db")
        messages = _make_messages()
        params = {"lambda_fact": 0.05, "lambda_episode": 0.2}

        result = prepare(
            db_path=db_path,
            messages=messages,
            params=params,
            embedding_provider=fake_provider,
            simulate_ticks=10,
        )

        assert result["status"] == "ok"
        assert result["ticks"] == 10
        assert result["cached"] is False

    def test_prepare_reuse_cached(self, tmp_path, fake_provider):
        db_path = str(tmp_path / "test.db")
        messages = _make_messages()
        params = {}

        # First call stores memories
        prepare(
            db_path=db_path,
            messages=messages,
            params=params,
            embedding_provider=fake_provider,
        )

        # Second call should detect existing memories and skip
        result = prepare(
            db_path=db_path,
            messages=messages,
            params=params,
            embedding_provider=fake_provider,
        )

        assert result["cached"] is True

    def test_prepare_with_custom_decay(self, tmp_path, fake_provider):
        db_path = str(tmp_path / "test.db")
        messages = _make_messages()

        # Identity decay: no change
        def identity_decay(activation, impact, stability, mtype, params):
            return activation

        result = prepare(
            db_path=db_path,
            messages=messages,
            params={},
            embedding_provider=fake_provider,
            simulate_ticks=5,
            custom_decay_fn=identity_decay,
        )

        assert result["status"] == "ok"


class TestSearch:
    def test_basic_search(self, tmp_path, fake_provider):
        db_path = str(tmp_path / "test.db")
        messages = _make_messages()

        prepare(
            db_path=db_path,
            messages=messages,
            params={},
            embedding_provider=fake_provider,
        )

        result = search_memories(
            db_path=db_path,
            query="what does the user like to bake?",
            embedding_provider=fake_provider,
            top_k=5,
        )

        assert "results" in result
        assert len(result["results"]) <= 5
        # All stored messages should be searchable
        assert len(result["results"]) == 3

        # Check result structure
        for r in result["results"]:
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "storage_score" in r
            assert "retrieval_score" in r
            assert "category" in r
            assert "created_tick" in r
            assert "speaker" in r

    def test_search_with_activation_weight(self, tmp_path, fake_provider):
        db_path = str(tmp_path / "test.db")
        messages = _make_messages()

        prepare(
            db_path=db_path,
            messages=messages,
            params={},
            embedding_provider=fake_provider,
            simulate_ticks=10,
        )

        result = search_memories(
            db_path=db_path,
            query="chocolate cake",
            embedding_provider=fake_provider,
            top_k=5,
            activation_weight=0.5,
        )

        assert "results" in result
        assert len(result["results"]) > 0

    def test_search_top_k_limit(self, tmp_path, fake_provider):
        db_path = str(tmp_path / "test.db")
        messages = _make_messages()

        prepare(
            db_path=db_path,
            messages=messages,
            params={},
            embedding_provider=fake_provider,
        )

        result = search_memories(
            db_path=db_path,
            query="cake",
            embedding_provider=fake_provider,
            top_k=1,
        )

        assert len(result["results"]) == 1


class TestCachedEmbeddingProvider:
    def test_caching(self, fake_provider_with_cache):
        provider = fake_provider_with_cache

        vec1 = provider.embed("hello world")
        vec2 = provider.embed("hello world")

        np.testing.assert_array_equal(vec1, vec2)
        # The raw provider should have been called only once
        assert provider._provider._call_count == 1

    def test_batch_caching(self, fake_provider_with_cache):
        provider = fake_provider_with_cache

        # First: cache one text
        provider.embed("already cached")

        # Batch: one cached + one new
        results = provider.embed_batch(["already cached", "brand new"])

        assert len(results) == 2
        # Raw provider called: 1 (first embed) + 1 (brand new in batch) = 2
        assert provider._provider._call_count == 2

    def test_no_cache_db(self):
        raw = FakeEmbeddingProvider(dim=64)
        provider = CachedEmbeddingProvider(provider=raw, model_name="fake")

        # Should work without a cache DB (no caching, just pass-through)
        vec = provider.embed("test")
        assert vec.shape == (64,)


class TestCLIArgParsing:
    def test_parser_prepare(self):
        from memory_decay.bench_bridge import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--action", "prepare",
            "--db-path", "/tmp/test.db",
            "--messages-file", "/tmp/msgs.json",
            "--params-file", "/tmp/params.json",
            "--simulate-ticks", "50",
            "--embedding-provider", "openai",
            "--embedding-api-key", "sk-test",
            "--embedding-model", "text-embedding-3-large",
            "--cache-db-path", "/tmp/cache.db",
        ])

        assert args.action == "prepare"
        assert args.db_path == "/tmp/test.db"
        assert args.simulate_ticks == 50
        assert args.cache_db_path == "/tmp/cache.db"

    def test_parser_search(self):
        from memory_decay.bench_bridge import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--action", "search",
            "--db-path", "/tmp/test.db",
            "--query", "what did I eat?",
            "--top-k", "20",
            "--activation-weight", "0.1",
            "--embedding-provider", "openai",
            "--embedding-api-key", "sk-test",
        ])

        assert args.action == "search"
        assert args.query == "what did I eat?"
        assert args.top_k == 20
        assert args.activation_weight == 0.1
