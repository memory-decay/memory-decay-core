"""Tests for building memorybench-compatible embedding caches."""

import json
import pickle

import numpy as np
import pytest

from scripts.build_memorybench_cache import build_embedding_cache, extract_texts, load_questions


FIXTURE = [
    {
        "question_id": "q1",
        "question": "What snack did I buy?",
        "haystack_sessions": [
            [
                {"role": "user", "content": "I bought pretzels."},
                {"role": "assistant", "content": "Pretzels are great."},
            ]
        ],
    },
    {
        "question_id": "q2",
        "question": "What snack did I buy?",
        "haystack_sessions": [
            [
                {"role": "user", "content": "I bought pretzels."},
                {"role": "assistant", "content": "Pretzels are great."},
            ]
        ],
    },
]


def test_load_questions_from_directory(tmp_path):
    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()
    for item in FIXTURE:
        (questions_dir / f"{item['question_id']}.json").write_text(json.dumps(item))

    loaded = load_questions(str(questions_dir))

    assert len(loaded) == 2
    assert {item["question_id"] for item in loaded} == {"q1", "q2"}


def test_extract_texts_includes_questions_and_prefixed_messages():
    texts = extract_texts(FIXTURE)

    assert "What snack did I buy?" in texts
    assert "[User] I bought pretzels." in texts
    assert "[Assistant] Pretzels are great." in texts


def test_extract_texts_deduplicates_repeated_content():
    texts = extract_texts(FIXTURE)

    assert texts.count("What snack did I buy?") == 1
    assert texts.count("[User] I bought pretzels.") == 1


def test_build_embedding_cache_writes_embeddings(tmp_path):
    output_dir = tmp_path / "cache"

    def batch_embedder(batch: list[str]) -> list[np.ndarray]:
        return [np.full(4, idx + 1, dtype=np.float32) for idx, _ in enumerate(batch)]

    build_embedding_cache(
        ["alpha", "beta"],
        str(output_dir),
        batch_embedder=batch_embedder,
        batch_size=2,
    )

    with open(output_dir / "embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    assert set(embeddings) == {"alpha", "beta"}
    assert np.array_equal(embeddings["alpha"], np.array([1, 1, 1, 1], dtype=np.float32))


def test_build_embedding_cache_resumes_from_existing_embeddings(tmp_path):
    output_dir = tmp_path / "cache"
    output_dir.mkdir()

    existing = {"alpha": np.array([9, 9, 9, 9], dtype=np.float32)}
    with open(output_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(existing, f)

    calls: list[list[str]] = []

    def batch_embedder(batch: list[str]) -> list[np.ndarray]:
        calls.append(batch)
        return [np.full(4, 1, dtype=np.float32) for _ in batch]

    build_embedding_cache(
        ["alpha", "beta", "gamma"],
        str(output_dir),
        batch_embedder=batch_embedder,
        batch_size=2,
    )

    with open(output_dir / "embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    assert np.array_equal(embeddings["alpha"], existing["alpha"])
    assert set(embeddings) == {"alpha", "beta", "gamma"}
    assert calls == [["beta", "gamma"]]


def test_build_embedding_cache_checkpoints_progress_before_failure(tmp_path):
    output_dir = tmp_path / "cache"

    calls: list[list[str]] = []

    def batch_embedder(batch: list[str]) -> list[np.ndarray]:
        calls.append(batch)
        if batch == ["gamma"]:
            raise RuntimeError("quota exceeded")
        return [np.full(4, 1, dtype=np.float32) for _ in batch]

    with pytest.raises(RuntimeError, match="quota exceeded"):
        build_embedding_cache(
            ["alpha", "beta", "gamma"],
            str(output_dir),
            batch_embedder=batch_embedder,
            batch_size=2,
            checkpoint_every=2,
        )

    with open(output_dir / "embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    assert set(embeddings) == {"alpha", "beta"}
    assert calls == [["alpha", "beta"], ["gamma"]]
