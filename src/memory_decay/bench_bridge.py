"""CLI bridge for MemoryBench → MemoryStore direct access.

Bypasses the HTTP server to store/search memories via SQLite directly.
Reduces per-question latency from ~70s (HTTP) to a few seconds.

Usage:
    python -m memory_decay.bench_bridge \
        --action prepare --db-path /tmp/conv123.db \
        --messages-file /tmp/messages.json \
        --params-file experiments/exp_bench_0001/params.json \
        --embedding-provider openai --embedding-api-key KEY \
        --embedding-model text-embedding-3-large \
        --simulate-ticks 100

    python -m memory_decay.bench_bridge \
        --action search --db-path /tmp/conv123.db \
        --query "what did I eat?" --top-k 30 \
        --embedding-provider openai --embedding-api-key KEY \
        --embedding-model text-embedding-3-large
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

from .decay import DecayEngine
from .embedding_provider import EmbeddingProvider, create_embedding_provider
from .memory_store import MemoryStore, _serialize_f32, _deserialize_f32
from .runner import _load_decay_fn


class CachedEmbeddingProvider:
    """Wraps an EmbeddingProvider with a shared SQLite embedding cache."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_db_path: str | None = None,
        model_name: str = "",
    ):
        self._provider = provider
        self._model_name = model_name
        self._cache_conn = None
        self._cache_db_path = cache_db_path
        if cache_db_path:
            self._init_cache(cache_db_path)

    def _init_cache(self, db_path: str) -> None:
        import sqlite3

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._cache_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._cache_conn.execute("PRAGMA journal_mode=WAL")
        self._cache_conn.execute(
            """CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                model     TEXT DEFAULT '',
                embedding BLOB NOT NULL
            )"""
        )
        self._cache_conn.commit()

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _get_cached(self, text: str) -> np.ndarray | None:
        if self._cache_conn is None:
            return None
        text_hash = self._hash(text)
        row = self._cache_conn.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model = ?",
            (text_hash, self._model_name),
        ).fetchone()
        if row is None:
            return None
        dim = self._provider.dimension
        return np.array(
            struct.unpack(f"{dim}f", row[0]), dtype=np.float32
        )

    def _put_cached(self, text: str, embedding: np.ndarray) -> None:
        if self._cache_conn is None:
            return
        text_hash = self._hash(text)
        blob = _serialize_f32(embedding)
        self._cache_conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, model, embedding) VALUES (?, ?, ?)",
            (text_hash, self._model_name, blob),
        )
        self._cache_conn.commit()

    def embed(self, text: str) -> np.ndarray:
        cached = self._get_cached(text)
        if cached is not None:
            return cached
        vec = self._provider.embed(text)
        self._put_cached(text, vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_vecs = self._provider.embed_batch(uncached_texts)
            for idx, vec in zip(uncached_indices, new_vecs):
                results[idx] = vec
                self._put_cached(texts[idx], vec)

        return results  # type: ignore[return-value]

    def close(self) -> None:
        if self._cache_conn:
            self._cache_conn.close()
            self._cache_conn = None


def prepare(
    db_path: str,
    messages: list[dict],
    params: dict,
    embedding_provider: CachedEmbeddingProvider,
    simulate_ticks: int = 0,
    custom_decay_fn=None,
) -> dict:
    """Store messages and run decay ticks. Returns status dict."""
    dim = embedding_provider.dimension
    store = MemoryStore(db_path, embedding_dim=dim)

    # Check if DB already has memories → reuse
    existing_count = store.num_memories
    if existing_count > 0:
        store.close()
        return {
            "status": "ok",
            "memories_count": existing_count,
            "ticks": simulate_ticks,
            "cached": True,
        }

    # Embed all messages
    texts = [m["text"] for m in messages]
    embeddings = embedding_provider.embed_batch(texts)

    # Store memories
    for i, (msg, emb) in enumerate(zip(messages, embeddings)):
        memory_id = f"m_{i:06d}"
        store.add_memory(
            memory_id=memory_id,
            content=msg["text"],
            embedding=emb,
            mtype=msg.get("mtype", "episode"),
            importance=msg.get("importance", 0.7),
            speaker=msg.get("speaker", ""),
            created_tick=msg.get("created_tick", 0),
        )

    memories_count = store.num_memories

    # Run decay ticks
    if simulate_ticks > 0:
        engine = DecayEngine(
            store=store,
            params=params,
            custom_decay_fn=custom_decay_fn,
        )
        for _ in range(simulate_ticks):
            engine.tick()

    store.close()
    return {
        "status": "ok",
        "memories_count": memories_count,
        "ticks": simulate_ticks,
        "cached": False,
    }


def search_memories(
    db_path: str,
    query: str,
    embedding_provider: CachedEmbeddingProvider,
    top_k: int = 30,
    activation_weight: float = 0.0,
) -> dict:
    """Search memories in existing DB. Returns results dict."""
    dim = embedding_provider.dimension
    store = MemoryStore(db_path, embedding_dim=dim)

    query_embedding = embedding_provider.embed(query)
    results = store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        activation_weight=activation_weight,
    )

    store.close()
    return {"results": results}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI bridge for MemoryBench → MemoryStore direct access"
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["prepare", "search"],
        help="Action to perform",
    )
    parser.add_argument("--db-path", required=True, help="Path to per-conversation SQLite DB")
    parser.add_argument("--messages-file", help="JSON file with messages (for prepare)")
    parser.add_argument("--params-file", help="JSON params file for decay engine")
    parser.add_argument("--query", help="Search query (for search)")
    parser.add_argument("--top-k", type=int, default=30, help="Number of search results")
    parser.add_argument("--simulate-ticks", type=int, default=0, help="Decay ticks to simulate")
    parser.add_argument("--activation-weight", type=float, default=None, help="Activation weight for search")
    parser.add_argument("--custom-decay-fn", help="Path to custom decay function file")
    parser.add_argument("--cache-db-path", help="Path to shared embedding cache DB")
    parser.add_argument("--embedding-provider", default="openai", help="Embedding provider name")
    parser.add_argument("--embedding-api-key", help="API key for embedding provider")
    parser.add_argument("--embedding-model", default="text-embedding-3-large", help="Embedding model name")
    parser.add_argument("--embedding-base-url", help="Base URL for embedding API")
    parser.add_argument("--embedding-dimensions", type=int, help="Embedding dimensions override")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load params
    params = {}
    activation_weight = 0.0
    if args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
        activation_weight = params.get("activation_weight", 0.0)

    if args.activation_weight is not None:
        activation_weight = args.activation_weight

    # Load custom decay function
    custom_decay_fn = None
    if args.custom_decay_fn:
        custom_decay_fn = _load_decay_fn(args.custom_decay_fn)

    # Create embedding provider
    api_key = args.embedding_api_key or os.environ.get("OPENAI_API_KEY", "")
    raw_provider = create_embedding_provider(
        provider=args.embedding_provider,
        api_key=api_key,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        dimensions=args.embedding_dimensions,
    )
    provider = CachedEmbeddingProvider(
        provider=raw_provider,
        cache_db_path=args.cache_db_path,
        model_name=args.embedding_model or "",
    )

    try:
        if args.action == "prepare":
            if not args.messages_file:
                print(json.dumps({"error": "--messages-file required for prepare"}))
                sys.exit(1)
            with open(args.messages_file) as f:
                messages = json.load(f)
            result = prepare(
                db_path=args.db_path,
                messages=messages,
                params=params,
                embedding_provider=provider,
                simulate_ticks=args.simulate_ticks,
                custom_decay_fn=custom_decay_fn,
            )
            print(json.dumps(result))

        elif args.action == "search":
            if not args.query:
                print(json.dumps({"error": "--query required for search"}))
                sys.exit(1)
            result = search_memories(
                db_path=args.db_path,
                query=args.query,
                embedding_provider=provider,
                top_k=args.top_k,
                activation_weight=activation_weight,
            )
            print(json.dumps(result))
    finally:
        provider.close()


if __name__ == "__main__":
    main()
