"""Integration test for the full auto-research experiment loop."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from memory_decay.cache_builder import build_cache, load_cache
from memory_decay.runner import run_experiment


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


EXPONENTIAL_FN = '''\
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 0.5)
    rho = params.get("stability_weight", 0.8)
    combined = max((1 + alpha * impact) * (1 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.02) if mtype == "fact" else params.get("lambda_episode", 0.035)
    return activation * math.exp(-lam / combined)
'''

POWER_LAW_FN = '''\
def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 0.5)
    rho = params.get("stability_weight", 0.8)
    combined = max((1 + alpha * impact) * (1 + rho * stability), 1e-9)
    beta = params.get("beta_fact", 0.08) if mtype == "fact" else params.get("beta_episode", 0.12)
    return activation / (1 + beta / combined)
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


class TestAutoResearchLoop:
    @pytest.fixture
    def setup(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)
        return tmp_path, str(cache_dir)

    def test_two_experiments_can_be_compared(self, setup):
        tmp_path, cache_dir = setup

        exp1 = tmp_path / "experiments" / "exp_0000"
        exp1.mkdir(parents=True)
        (exp1 / "decay_fn.py").write_text(EXPONENTIAL_FN)
        (exp1 / "params.json").write_text(json.dumps(DEFAULT_PARAMS))
        (exp1 / "hypothesis.txt").write_text("baseline exponential")

        result1 = run_experiment(str(exp1), cache_dir, total_ticks=50, eval_interval=10)
        assert result1["status"] == "completed"

        exp2 = tmp_path / "experiments" / "exp_0001"
        exp2.mkdir(parents=True)
        (exp2 / "decay_fn.py").write_text(POWER_LAW_FN)
        (exp2 / "params.json").write_text(json.dumps(DEFAULT_PARAMS))
        (exp2 / "hypothesis.txt").write_text("power law has longer tail")

        result2 = run_experiment(str(exp2), cache_dir, total_ticks=50, eval_interval=10)
        assert result2["status"] == "completed"

        assert 0.0 <= result1["overall_score"] <= 1.0
        assert 0.0 <= result2["overall_score"] <= 1.0

    def test_history_tracking(self, setup):
        tmp_path, cache_dir = setup

        exp_dir = tmp_path / "experiments" / "exp_0000"
        exp_dir.mkdir(parents=True)
        (exp_dir / "decay_fn.py").write_text(EXPONENTIAL_FN)
        (exp_dir / "params.json").write_text(json.dumps(DEFAULT_PARAMS))
        (exp_dir / "hypothesis.txt").write_text("baseline")

        result = run_experiment(str(exp_dir), cache_dir, total_ticks=50, eval_interval=10)

        history_path = tmp_path / "experiments" / "history.jsonl"
        entry = {
            "exp": "exp_0000",
            "overall": result["overall_score"],
            "retrieval": result["retrieval_score"],
            "plausibility": result["plausibility_score"],
            "status": "baseline",
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        with open(history_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["exp"] == "exp_0000"
