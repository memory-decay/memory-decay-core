"""Pre-compute and cache embeddings for the auto-research loop.

Caches all memory content texts and recall query texts so that
repeated simulation runs require zero embedding API calls.

Note: Uses pickle for numpy array serialization. The cache is always
self-generated from the local dataset — never loaded from untrusted sources.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def build_cache(
    dataset_path: str,
    cache_dir: str,
    embedder: Optional[Callable] = None,
    embedding_backend: str = "auto",
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

    embeddings: dict[str, np.ndarray] = {}
    texts = set()

    for item in dataset:
        texts.add(item["content"])
        if "recall_query" in item:
            texts.add(item["recall_query"])

    for i, text in enumerate(sorted(texts)):
        emb = np.array(embedder(text), dtype=np.float32)
        embeddings[text] = emb
        if (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{len(texts)} texts")

    print(f"  Cached {len(embeddings)} embeddings")

    with open(cache_path / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    with open(cache_path / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    test_queries = [
        (item["recall_query"], item["id"])
        for item in dataset
        if "recall_query" in item
    ]
    with open(cache_path / "test_queries.json", "w", encoding="utf-8") as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)


def load_cache(cache_dir: str) -> tuple[Callable, list[dict], list[tuple[str, str]]]:
    cache_path = Path(cache_dir)

    with open(cache_path / "embeddings.pkl", "rb") as f:
        embeddings: dict[str, np.ndarray] = pickle.load(f)

    with open(cache_path / "dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(cache_path / "test_queries.json", "r", encoding="utf-8") as f:
        test_queries = [tuple(q) for q in json.load(f)]

    def cached_embedder(text: str) -> np.ndarray:
        if text not in embeddings:
            raise KeyError(f"Text not in embedding cache: {text[:80]}...")
        return embeddings[text].copy()

    return cached_embedder, dataset, test_queries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build embedding cache")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", default="cache", help="Cache output directory")
    parser.add_argument(
        "--backend", choices=["auto", "local", "gemini"], default="auto"
    )
    args = parser.parse_args()

    build_cache(args.dataset, args.output, embedding_backend=args.backend)
    print(f"Cache saved to {args.output}/")
