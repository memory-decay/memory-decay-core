"""Pluggable embedding provider abstraction.

Supports Gemini and OpenAI embedding APIs with a common interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts. Override for batch API support."""
        return [self.embed(t) for t in texts]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embedding provider."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self._api_key = api_key
        self._model = model
        self._client = None
        self._dim = 768

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        self._ensure_client()
        result = self._client.models.embed_content(
            model=self._model, contents=text,
        )
        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        self._dim = vec.shape[0]
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self._ensure_client()
        result = self._client.models.embed_content(
            model=self._model, contents=texts,
        )
        vecs = [np.array(e.values, dtype=np.float32) for e in result.embeddings]
        if vecs:
            self._dim = vecs[0].shape[0]
        return vecs


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding provider."""

    KNOWN_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        dimensions: int | None = None,
    ):
        import openai
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dimensions = dimensions
        self._dim = dimensions or self.KNOWN_DIMS.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        params: dict = {"model": self._model, "input": text}
        if self._dimensions:
            params["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**params)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        params: dict = {"model": self._model, "input": texts}
        if self._dimensions:
            params["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**params)
        return [
            np.array(item.embedding, dtype=np.float32)
            for item in sorted(response.data, key=lambda x: x.index)
        ]


def create_embedding_provider(
    provider: str,
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
    dimensions: int | None = None,
) -> EmbeddingProvider:
    """Factory function to create an embedding provider."""
    if provider == "gemini":
        return GeminiEmbeddingProvider(
            api_key=api_key,
            model=model or "gemini-embedding-001",
        )
    elif provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            base_url=base_url,
            dimensions=dimensions,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'openai'.")
