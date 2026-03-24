"""Tests for pluggable embedding providers."""

import numpy as np
import pytest

from memory_decay.embedding_provider import (
    EmbeddingProvider,
    create_embedding_provider,
)


class FakeProvider(EmbeddingProvider):
    """Deterministic provider for testing."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % 2**31)
        return rng.randn(self._dim).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


class TestEmbeddingProvider:
    def test_create_provider_local_default(self):
        p = create_embedding_provider(provider="local")
        assert p.dimension == 768

    def test_fake_provider_returns_correct_dim(self):
        p = FakeProvider(dim=16)
        vec = p.embed("hello")
        assert vec.shape == (16,)

    def test_fake_provider_deterministic(self):
        p = FakeProvider()
        v1 = p.embed("test")
        v2 = p.embed("test")
        np.testing.assert_array_equal(v1, v2)

    def test_embed_batch(self):
        p = FakeProvider()
        results = p.embed_batch(["a", "b"])
        assert len(results) == 2
        assert results[0].shape == (8,)

    def test_create_provider_gemini(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        p = create_embedding_provider(
            provider="gemini",
            api_key="fake-key",
        )
        assert p.dimension == 768

    def test_create_provider_openai(self):
        p = create_embedding_provider(
            provider="openai",
            api_key="fake-key",
            model="text-embedding-3-small",
        )
        assert p.dimension == 1536

    def test_create_provider_gemini_missing_key_raises(self):
        with pytest.raises(ValueError, match="requires an API key"):
            create_embedding_provider(provider="gemini")

    def test_create_provider_openai_missing_key_raises(self):
        with pytest.raises(ValueError, match="requires an API key"):
            create_embedding_provider(provider="openai")

    def test_create_provider_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_embedding_provider(provider="unknown", api_key="x")
