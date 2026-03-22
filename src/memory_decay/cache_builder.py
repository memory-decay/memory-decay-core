"""Pre-compute and cache embeddings for the auto-research loop.

Caches all memory content texts and recall query texts so that
repeated simulation runs require zero embedding API calls.

Note: Uses pickle for numpy array serialization. The cache is always
self-generated from the local dataset — never loaded from untrusted sources.
"""

from __future__ import annotations

import json
import pickle
import random as _rnd
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def build_cache(
    dataset_path: str,
    cache_dir: str,
    embedder: Optional[Callable] = None,
    batch_embedder: Optional[Callable] = None,
    embedding_backend: str = "auto",
    test_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 100,
) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    if embedder is None:
        from .graph import MemoryGraph
        graph = MemoryGraph(embedding_backend=embedding_backend)
        embedder = graph._embed_text
        if batch_embedder is None:
            batch_embedder = graph._gemini_embed_batch

    embeddings: dict[str, np.ndarray] = {}
    texts = set()

    for item in dataset:
        texts.add(item["content"])
        if "recall_query" in item:
            texts.add(item["recall_query"])

    sorted_texts = sorted(texts)

    if batch_embedder is not None:
        for batch_start in range(0, len(sorted_texts), batch_size):
            batch = sorted_texts[batch_start:batch_start + batch_size]
            batch_embeddings = batch_embedder(batch)
            for text, emb in zip(batch, batch_embeddings):
                embeddings[text] = np.array(emb, dtype=np.float32)
            processed = min(batch_start + batch_size, len(sorted_texts))
            print(f"  Embedded {processed}/{len(sorted_texts)} texts")
    else:
        for i, text in enumerate(sorted_texts):
            emb = np.array(embedder(text), dtype=np.float32)
            embeddings[text] = emb
            if (i + 1) % 50 == 0:
                print(f"  Embedded {i + 1}/{len(sorted_texts)} texts")

    print(f"  Cached {len(embeddings)} embeddings")

    with open(cache_path / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    with open(cache_path / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # --- train/test split (stratified by type) ---
    rng = _rnd.Random(seed)
    facts = [item for item in dataset if item.get("type") == "fact"]
    episodes = [item for item in dataset if item.get("type") == "episode"]
    rng.shuffle(facts)
    rng.shuffle(episodes)

    n_test_facts = max(1, int(len(facts) * test_ratio))
    n_test_episodes = max(1, int(len(episodes) * test_ratio)) if episodes else 0
    test_items = facts[:n_test_facts] + episodes[:n_test_episodes]
    train_items = facts[n_test_facts:] + episodes[n_test_episodes:]

    test_queries = [
        (item["recall_query"], item["id"])
        for item in test_items if "recall_query" in item
    ]
    rehearsal_targets = [item["id"] for item in train_items]

    with open(cache_path / "test_queries.json", "w", encoding="utf-8") as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)

    with open(cache_path / "rehearsal_targets.json", "w", encoding="utf-8") as f:
        json.dump(rehearsal_targets, f, ensure_ascii=False, indent=2)


def load_cache(
    cache_dir: str,
    fallback_embedder: Optional[Callable] = None,
) -> tuple[Callable, list[dict], list[tuple[str, str]], list[str]]:
    """Load embedding cache with optional fallback to live encoding.

    If fallback_embedder is provided, texts not in the cache will be
    encoded live and cached for subsequent calls.
    """
    cache_path = Path(cache_dir)

    with open(cache_path / "embeddings.pkl", "rb") as f:
        embeddings: dict[str, np.ndarray] = pickle.load(f)

    with open(cache_path / "dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(cache_path / "test_queries.json", "r", encoding="utf-8") as f:
        test_queries = [tuple(q) for q in json.load(f)]

    with open(cache_path / "rehearsal_targets.json", "r", encoding="utf-8") as f:
        rehearsal_targets = json.load(f)

    def cached_embedder(text: str) -> np.ndarray:
        if text not in embeddings:
            if fallback_embedder is not None:
                emb = np.array(fallback_embedder(text), dtype=np.float32)
                embeddings[text] = emb
                return emb.copy()
            raise KeyError(f"Text not in embedding cache: {text[:80]}...")
        return embeddings[text].copy()

    return cached_embedder, dataset, test_queries, rehearsal_targets


def load_raw_dataset(path: Path) -> list[dict]:
    """Load the raw dataset without train/test splitting."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cached_embedder(
    cache_dir: str,
    fallback_embedder: Optional[Callable] = None,
) -> Callable:
    """Load the cached embedder function from the cache directory.

    If fallback_embedder is provided, texts not in the cache will be
    encoded live and cached for subsequent calls.

    Note: Uses pickle for numpy array deserialization. The cache is always
    self-generated from the local dataset — never loaded from untrusted sources.
    """
    cache_path = Path(cache_dir)

    with open(cache_path / "embeddings.pkl", "rb") as f:
        embeddings: dict[str, np.ndarray] = pickle.load(f)  # noqa: S301

    def cached_embedder(text: str) -> np.ndarray:
        if text not in embeddings:
            if fallback_embedder is not None:
                emb = np.array(fallback_embedder(text), dtype=np.float32)
                embeddings[text] = emb
                return emb.copy()
            raise KeyError(f"Text not in embedding cache: {text[:80]}...")
        return embeddings[text].copy()

    return cached_embedder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build embedding cache")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", default="cache", help="Cache output directory")
    parser.add_argument(
        "--backend", choices=["auto", "local", "gemini"], default="auto"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Texts per batch API call (default: 100)"
    )
    args = parser.parse_args()

    build_cache(args.dataset, args.output, embedding_backend=args.backend, batch_size=args.batch_size)
    print(f"Cache saved to {args.output}/")
