"""Tests for embedding cache builder."""

import json
import pickle

import numpy as np
import pytest

from memory_decay.cache_builder import build_cache, load_cache


SAMPLE_DATASET = [
    {
        "id": "f1", "type": "fact",
        "content": "서울은 대한민국의 수도이다",
        "entities": ["서울"], "tick": 0, "impact": 0.9,
        "associations": [],
        "recall_query": "대한민국의 수도는?", "recall_answer": "서울",
    },
    {
        "id": "e1", "type": "episode",
        "content": "커피를 마셨다",
        "entities": ["커피"], "tick": 5, "impact": 0.5,
        "associations": [{"id": "f1", "weight": 0.6}],
        "recall_query": "무엇을 마셨는가?", "recall_answer": "커피",
    },
]


def mock_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(384).astype(np.float32)


class TestCacheBuilder:
    def test_build_cache_creates_files(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)

        assert (cache_dir / "embeddings.pkl").exists()
        assert (cache_dir / "dataset.json").exists()
        assert (cache_dir / "test_queries.json").exists()

    def test_cache_contains_all_texts(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)

        with open(cache_dir / "embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)

        for item in SAMPLE_DATASET:
            assert item["content"] in embeddings
            assert isinstance(embeddings[item["content"]], np.ndarray)

        for item in SAMPLE_DATASET:
            if "recall_query" in item:
                assert item["recall_query"] in embeddings

    def test_build_cache_creates_rehearsal_targets(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)

        assert (cache_dir / "rehearsal_targets.json").exists()

    def test_load_cache_returns_embedder(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)

        cached_embedder, dataset, test_queries, rehearsal_targets = load_cache(str(cache_dir))

        emb = cached_embedder("서울은 대한민국의 수도이다")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)

        assert len(dataset) == 2
        assert isinstance(rehearsal_targets, list)

    def test_load_cache_raises_for_unknown_text(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)

        cached_embedder, _, _, _ = load_cache(str(cache_dir))

        with pytest.raises(KeyError):
            cached_embedder("이건 캐시에 없는 텍스트")


def _fixed_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def test_cache_splits_train_test(tmp_path):
    """Cache must produce separate test_queries and rehearsal_targets."""
    dataset = [
        {"id": f"m{i}", "type": "fact", "content": f"fact {i}",
         "tick": i * 10, "impact": 0.5, "associations": [],
         "recall_query": f"query {i}"}
        for i in range(10)
    ]
    dataset_path = tmp_path / "data.jsonl"
    with open(dataset_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    build_cache(str(dataset_path), str(tmp_path / "cache"), embedder=_fixed_embedder)
    _, full_dataset, test_queries, rehearsal_targets = load_cache(str(tmp_path / "cache"))

    test_ids = {tid for _, tid in test_queries}
    rehearsal_ids = set(rehearsal_targets)

    # No overlap
    assert test_ids.isdisjoint(rehearsal_ids), (
        f"Overlap between test and rehearsal: {test_ids & rehearsal_ids}"
    )

    # Together they cover the full dataset
    assert test_ids | rehearsal_ids == {item["id"] for item in full_dataset}
