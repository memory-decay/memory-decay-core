"""Pluggable embedding provider abstraction.

Supports Gemini and OpenAI embedding APIs with a common interface.
"""

from __future__ import annotations

import asyncio
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

    async def aembed(self, text: str) -> np.ndarray:
        """Async embed. Default wraps sync via to_thread."""
        return await asyncio.to_thread(self.embed, text)

    async def aembed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Async batch embed. Default wraps sync via to_thread."""
        return await asyncio.to_thread(self.embed_batch, texts)


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embedding provider."""

    KNOWN_DIMS = {
        "gemini-embedding-001": 3072,
        "text-embedding-004": 768,
    }

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self._api_key = api_key
        self._model = model
        self._client = None
        self._dim = self.KNOWN_DIMS.get(model, 3072)

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


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""

    KNOWN_DIMS = {
        "jhgan/ko-sroberta-multitask": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "intfloat/multilingual-e5-large": 1024,
    }

    def __init__(self, model: str = "jhgan/ko-sroberta-multitask"):
        self._model_name = model
        self._st_model = None
        self._dim = self.KNOWN_DIMS.get(model, 768)

    def _ensure_model(self):
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings but failed to import.\n"
                    f"Underlying error: {e}\n\n"
                    "Install with: pip install 'memory-decay[local]'\n\n"
                    "If already installed and using Python 3.13.x, note that CPython 3.13.8 "
                    "has an ast.parse() regression that breaks torch (pytorch/pytorch#178255).\n"
                    "Fix: upgrade to Python 3.13.11+ or use Python 3.10-3.12."
                ) from e
            self._st_model = SentenceTransformer(self._model_name)
            self._dim = self._st_model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        self._ensure_model()
        return np.array(self._st_model.encode(text), dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self._ensure_model()
        embeddings = self._st_model.encode(texts)
        return [np.array(e, dtype=np.float32) for e in embeddings]


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
        self._async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dimensions = dimensions
        self._dim = dimensions or self.KNOWN_DIMS.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed_params(self, input_val: str | list[str]) -> dict:
        params: dict = {"model": self._model, "input": input_val}
        if self._dimensions:
            params["dimensions"] = self._dimensions
        return params

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(**self._embed_params(text))
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        response = self._client.embeddings.create(**self._embed_params(texts))
        return [
            np.array(item.embedding, dtype=np.float32)
            for item in sorted(response.data, key=lambda x: x.index)
        ]

    async def aembed(self, text: str) -> np.ndarray:
        response = await self._async_client.embeddings.create(**self._embed_params(text))
        return np.array(response.data[0].embedding, dtype=np.float32)

    async def aembed_batch(self, texts: list[str]) -> list[np.ndarray]:
        response = await self._async_client.embeddings.create(**self._embed_params(texts))
        return [
            np.array(item.embedding, dtype=np.float32)
            for item in sorted(response.data, key=lambda x: x.index)
        ]


def create_embedding_provider(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    dimensions: int | None = None,
) -> EmbeddingProvider:
    """Factory function to create an embedding provider."""
    if provider == "local":
        return LocalEmbeddingProvider(model=model or "jhgan/ko-sroberta-multitask")

    if provider == "gemini":
        if not api_key:
            raise ValueError("Gemini provider requires an API key.")
        return GeminiEmbeddingProvider(
            api_key=api_key,
            model=model or "gemini-embedding-001",
        )

    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI provider requires an API key.")
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            base_url=base_url,
            dimensions=dimensions,
        )

    raise ValueError(f"Unknown provider: {provider}. Use 'local', 'gemini', or 'openai'.")
