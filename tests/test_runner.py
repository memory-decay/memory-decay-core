"""Tests for the single-experiment runner."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from memory_decay.runner import run_experiment, validate_decay_fn


SAMPLE_DATASET = [
    {"id": "f1", "type": "fact", "content": "서울은 대한민국의 수도이다",
     "entities": ["서울"], "tick": 0, "impact": 0.9, "associations": [],
     "recall_query": "대한민국의 수도는?", "recall_answer": "서울"},
    {"id": "f2", "type": "fact", "content": "김민수는 커피를 좋아한다",
     "entities": ["김민수"], "tick": 5, "impact": 0.7,
     "associations": [{"id": "f1", "weight": 0.6}],
     "recall_query": "김민수는 무엇을 좋아하는가?", "recall_answer": "커피"},
    {"id": "e1", "type": "episode", "content": "서울에서 커피를 마셨다",
     "entities": ["서울"], "tick": 10, "impact": 0.5,
     "associations": [{"id": "f1", "weight": 0.8}],
     "recall_query": "어디서 커피를 마셨는가?", "recall_answer": "서울"},
    {"id": "e2", "type": "episode", "content": "길에서 강아지를 만났다",
     "entities": ["강아지"], "tick": 20, "impact": 0.3, "associations": [],
     "recall_query": "길에서 만난 동물은?", "recall_answer": "강아지"},
]


def mock_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(384).astype(np.float32)


def build_test_cache(cache_dir: Path):
    """Build a test cache with self-generated embeddings.

    Note: Uses pickle for numpy array serialization, matching the
    cache_builder module. Cache is always self-generated from the
    local dataset -- never loaded from untrusted sources.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings = {}
    for item in SAMPLE_DATASET:
        embeddings[item["content"]] = mock_embedder(item["content"])
        if "recall_query" in item:
            embeddings[item["recall_query"]] = mock_embedder(item["recall_query"])
    with open(cache_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    with open(cache_dir / "dataset.json", "w") as f:
        json.dump(SAMPLE_DATASET, f, ensure_ascii=False)
    test_items = SAMPLE_DATASET[:1]
    train_items = SAMPLE_DATASET[1:]
    with open(cache_dir / "test_queries.json", "w") as f:
        json.dump([(item["recall_query"], item["id"]) for item in test_items], f)
    with open(cache_dir / "rehearsal_targets.json", "w") as f:
        json.dump([item["id"] for item in train_items], f)


GOOD_DECAY_FN = '''\
import math

def compute_decay(activation, impact, stability, mtype, params):
    lam = params.get("lambda_fact", 0.02) if mtype == "fact" else params.get("lambda_episode", 0.035)
    alpha = params.get("alpha", 0.5)
    rho = params.get("stability_weight", 0.8)
    effective = lam / max((1 + alpha * impact) * (1 + rho * stability), 1e-9)
    return activation * math.exp(-effective)
'''

BAD_DECAY_RETURNS_CONSTANT = '''\
def compute_decay(activation, impact, stability, mtype, params):
    return 1.0  # No decay at all
'''

BAD_DECAY_SYNTAX_ERROR = '''\
def compute_decay(activation, impact, stability, mtype, params)
    return 0.5
'''

DEFAULT_PARAMS = {
    "lambda_fact": 0.02, "lambda_episode": 0.035,
    "beta_fact": 0.08, "beta_episode": 0.12,
    "alpha": 0.5, "stability_weight": 0.8,
    "stability_decay": 0.01,
    "reinforcement_gain_direct": 0.2,
    "reinforcement_gain_assoc": 0.05,
    "stability_cap": 1.0,
}


class TestValidateDecayFn:
    def test_valid_function_passes(self, tmp_path):
        fn_path = tmp_path / "decay_fn.py"
        fn_path.write_text(GOOD_DECAY_FN)

        ok, error = validate_decay_fn(str(fn_path), DEFAULT_PARAMS)
        assert ok is True
        assert error is None

    def test_syntax_error_caught(self, tmp_path):
        fn_path = tmp_path / "decay_fn.py"
        fn_path.write_text(BAD_DECAY_SYNTAX_ERROR)

        ok, error = validate_decay_fn(str(fn_path), {})
        assert ok is False
        assert "syntax" in error.lower() or "SyntaxError" in error

    def test_no_decay_function_caught(self, tmp_path):
        fn_path = tmp_path / "decay_fn.py"
        fn_path.write_text(BAD_DECAY_RETURNS_CONSTANT)

        ok, error = validate_decay_fn(str(fn_path), {})
        assert ok is False
        assert "decay" in error.lower() or "constant" in error.lower()


class TestRunExperiment:
    def test_successful_experiment(self, tmp_path):
        cache_dir = tmp_path / "cache"
        build_test_cache(cache_dir)

        exp_dir = tmp_path / "exp_0001"
        exp_dir.mkdir()
        (exp_dir / "decay_fn.py").write_text(GOOD_DECAY_FN)
        (exp_dir / "params.json").write_text(json.dumps(DEFAULT_PARAMS))
        (exp_dir / "hypothesis.txt").write_text("baseline exponential test")

        run_experiment(str(exp_dir), str(cache_dir))

        results_path = exp_dir / "results.json"
        assert results_path.exists()

        results = json.loads(results_path.read_text())
        assert results["status"] == "completed"
        assert "overall_score" in results
        assert 0.0 <= results["overall_score"] <= 1.0

    def test_validation_failure_recorded(self, tmp_path):
        cache_dir = tmp_path / "cache"
        build_test_cache(cache_dir)

        exp_dir = tmp_path / "exp_bad"
        exp_dir.mkdir()
        (exp_dir / "decay_fn.py").write_text(BAD_DECAY_SYNTAX_ERROR)
        (exp_dir / "params.json").write_text("{}")
        (exp_dir / "hypothesis.txt").write_text("bad syntax test")

        run_experiment(str(exp_dir), str(cache_dir))

        results = json.loads((exp_dir / "results.json").read_text())
        assert results["status"] == "validation_failed"
        assert "error" in results
