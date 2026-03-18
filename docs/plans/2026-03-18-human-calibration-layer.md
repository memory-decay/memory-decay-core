# Human Calibration Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a human-review calibration pipeline that fits `fact` memory decay parameters on real review logs, then reuses the fitted parameters in the existing synthetic graph benchmark.

**Architecture:** Add a parallel evaluation stack beside the current retrieval evaluator. Real review logs are normalized into a common event schema, replayed through the existing `DecayEngine` state dynamics, and scored with probabilistic calibration metrics. A small optimizer searches `fact`-side parameters against held-out human events, then a bridge runner executes the existing synthetic benchmark with those fitted parameters for external validation.

**Tech Stack:** Python 3.10+, existing `memory_decay` package, `pytest`, `numpy`, `json`, existing `uv` workflow

---

### Task 1: Add normalized human review event schema

**Files:**
- Create: `src/memory_decay/human_data.py`
- Test: `tests/test_human_data.py`

**Step 1: Write the failing test**

Create `tests/test_human_data.py`:

```python
import json

from memory_decay.human_data import (
    load_review_events_jsonl,
    normalize_duolingo_event,
    normalize_anki_event,
)


def test_normalize_duolingo_event_maps_binary_outcome():
    raw = {
        "user_id": "u1",
        "lexeme_id": "bonjour<abc>",
        "delta": 2.5,
        "history_correct": 3,
        "history_seen": 5,
        "p_recall": 1.0,
    }

    event = normalize_duolingo_event(raw)

    assert event["user_id"] == "u1"
    assert event["item_id"] == "bonjour<abc>"
    assert event["memory_type"] == "fact"
    assert event["t_elapsed"] == 2.5
    assert event["review_index"] == 5
    assert event["outcome"] == 1
    assert event["grade"] is None


def test_normalize_anki_event_maps_grade_to_binary_outcome():
    raw = {
        "user_id": "u2",
        "card_id": "c9",
        "elapsed_days": 7,
        "review_th": 4,
        "rating": 1,
    }

    event = normalize_anki_event(raw)

    assert event["user_id"] == "u2"
    assert event["item_id"] == "c9"
    assert event["memory_type"] == "fact"
    assert event["t_elapsed"] == 7.0
    assert event["review_index"] == 4
    assert event["outcome"] == 0
    assert event["grade"] == 1


def test_load_review_events_jsonl_filters_invalid_rows(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [
        {"user_id": "u1", "item_id": "i1", "memory_type": "fact", "t_elapsed": 1.0, "review_index": 1, "outcome": 1, "grade": None, "metadata": {}},
        {"user_id": "u1", "item_id": "i2"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    events = load_review_events_jsonl(path)

    assert len(events) == 1
    assert events[0]["item_id"] == "i1"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory_decay.human_data'`

**Step 3: Write minimal implementation**

Create `src/memory_decay/human_data.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


REQUIRED_KEYS = {
    "user_id",
    "item_id",
    "memory_type",
    "t_elapsed",
    "review_index",
    "outcome",
    "grade",
    "metadata",
}


def normalize_duolingo_event(raw: dict) -> dict:
    return {
        "user_id": str(raw["user_id"]),
        "item_id": str(raw["lexeme_id"]),
        "memory_type": "fact",
        "t_elapsed": float(raw["delta"]),
        "review_index": int(raw["history_seen"]),
        "outcome": int(float(raw["p_recall"]) >= 0.5),
        "grade": None,
        "metadata": {
            "history_correct": int(raw.get("history_correct", 0)),
            "history_seen": int(raw.get("history_seen", 0)),
        },
    }


def normalize_anki_event(raw: dict) -> dict:
    rating = int(raw["rating"])
    return {
        "user_id": str(raw["user_id"]),
        "item_id": str(raw["card_id"]),
        "memory_type": "fact",
        "t_elapsed": float(raw["elapsed_days"]),
        "review_index": int(raw["review_th"]),
        "outcome": 0 if rating == 1 else 1,
        "grade": rating,
        "metadata": {},
    }


def load_review_events_jsonl(path: str | Path) -> list[dict]:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if REQUIRED_KEYS.issubset(row):
                events.append(row)
    return events
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_data.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
cd /Users/lit/memory-decay
git add src/memory_decay/human_data.py tests/test_human_data.py
git commit -m "Add normalized human review event schema"
```

---

### Task 2: Add event replay state model and probabilistic observation model

**Files:**
- Create: `src/memory_decay/human_eval.py`
- Test: `tests/test_human_eval.py`

**Step 1: Write the failing test**

Create `tests/test_human_eval.py`:

```python
import math

from memory_decay.human_eval import (
    HumanCalibrationEvaluator,
    sigmoid_probability,
)


def test_sigmoid_probability_is_bounded():
    assert 0.0 < sigmoid_probability(-100) < 0.01
    assert 0.99 < sigmoid_probability(100) < 1.0


def test_replay_event_returns_probability_and_updates_state():
    evaluator = HumanCalibrationEvaluator(
        decay_params={
            "lambda_fact": 0.02,
            "lambda_episode": 0.035,
            "beta_fact": 0.08,
            "beta_episode": 0.12,
            "alpha": 0.5,
            "stability_weight": 0.8,
            "stability_decay": 0.01,
            "reinforcement_gain_direct": 0.2,
            "reinforcement_gain_assoc": 0.05,
            "stability_cap": 1.0,
        },
        observation_params={"activation_scale": 6.0, "bias": -3.0},
    )
    event = {
        "user_id": "u1",
        "item_id": "i1",
        "memory_type": "fact",
        "t_elapsed": 3.0,
        "review_index": 1,
        "outcome": 1,
        "grade": None,
        "metadata": {},
    }

    result = evaluator.replay_event(event)

    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert result["activation_before_review"] < 1.0
    state = evaluator.get_state("u1", "i1")
    assert state["stability"] > 0.0


def test_metrics_include_nll_brier_and_ece():
    evaluator = HumanCalibrationEvaluator(
        decay_params={"lambda_fact": 0.02, "lambda_episode": 0.035, "beta_fact": 0.08, "beta_episode": 0.12, "alpha": 0.5, "stability_weight": 0.8, "stability_decay": 0.01, "reinforcement_gain_direct": 0.2, "reinforcement_gain_assoc": 0.05, "stability_cap": 1.0},
        observation_params={"activation_scale": 6.0, "bias": -3.0},
    )
    events = [
        {"user_id": "u1", "item_id": "i1", "memory_type": "fact", "t_elapsed": 1.0, "review_index": 1, "outcome": 1, "grade": None, "metadata": {}},
        {"user_id": "u1", "item_id": "i1", "memory_type": "fact", "t_elapsed": 12.0, "review_index": 2, "outcome": 0, "grade": None, "metadata": {}},
    ]

    metrics = evaluator.evaluate(events)

    assert set(metrics) >= {"nll", "brier", "ece", "num_events"}
    assert metrics["num_events"] == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory_decay.human_eval'`

**Step 3: Write minimal implementation**

Create `src/memory_decay/human_eval.py`:

```python
from __future__ import annotations

import math
import numpy as np


def sigmoid_probability(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(logit, 60.0), -60.0)))


class HumanCalibrationEvaluator:
    def __init__(self, decay_params: dict, observation_params: dict):
        self.decay_params = dict(decay_params)
        self.observation_params = dict(observation_params)
        self._state: dict[tuple[str, str], dict] = {}

    def get_state(self, user_id: str, item_id: str) -> dict:
        return self._state[(user_id, item_id)]

    def _ensure_state(self, event: dict) -> dict:
        key = (event["user_id"], event["item_id"])
        if key not in self._state:
            self._state[key] = {
                "activation": 1.0,
                "stability": 0.0,
                "memory_type": event["memory_type"],
            }
        return self._state[key]

    def _decay_once(self, activation: float, stability: float, memory_type: str, t_elapsed: float) -> float:
        lam = self.decay_params["lambda_fact"] if memory_type == "fact" else self.decay_params["lambda_episode"]
        rho = self.decay_params["stability_weight"]
        effective = lam / max(1.0 + rho * stability, 1e-9)
        return activation * math.exp(-effective * t_elapsed)

    def replay_event(self, event: dict) -> dict:
        state = self._ensure_state(event)
        activation_before = self._decay_once(
            state["activation"],
            state["stability"],
            state["memory_type"],
            float(event["t_elapsed"]),
        )
        logit = (
            self.observation_params["activation_scale"] * activation_before
            + self.observation_params.get("stability_scale", 0.0) * state["stability"]
            + self.observation_params["bias"]
        )
        prob = sigmoid_probability(logit)
        if int(event["outcome"]) == 1:
            gain = self.decay_params["reinforcement_gain_direct"]
            cap = self.decay_params["stability_cap"]
            state["stability"] = min(cap, state["stability"] + gain * (1.0 - state["stability"] / cap))
            state["activation"] = 1.0
        else:
            state["activation"] = activation_before
        return {
            "predicted_probability": prob,
            "activation_before_review": activation_before,
            "outcome": int(event["outcome"]),
        }

    def evaluate(self, events: list[dict]) -> dict:
        preds = []
        labels = []
        for event in events:
            result = self.replay_event(event)
            preds.append(min(max(result["predicted_probability"], 1e-6), 1 - 1e-6))
            labels.append(result["outcome"])
        preds_arr = np.array(preds)
        labels_arr = np.array(labels)
        nll = float(-(labels_arr * np.log(preds_arr) + (1 - labels_arr) * np.log(1 - preds_arr)).mean())
        brier = float(np.mean((preds_arr - labels_arr) ** 2))
        ece = float(abs(preds_arr.mean() - labels_arr.mean()))
        return {"nll": nll, "brier": brier, "ece": ece, "num_events": len(events)}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_eval.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
cd /Users/lit/memory-decay
git add src/memory_decay/human_eval.py tests/test_human_eval.py
git commit -m "Add human event replay evaluator"
```

---

### Task 3: Add train/validation/test split without temporal leakage

**Files:**
- Modify: `src/memory_decay/human_data.py`
- Create: `tests/test_human_split.py`

**Step 1: Write the failing test**

Create `tests/test_human_split.py`:

```python
from memory_decay.human_data import split_review_events


def test_split_review_events_keeps_users_disjoint():
    events = []
    for user_id in ["u1", "u2", "u3", "u4", "u5"]:
        for idx in range(3):
            events.append({
                "user_id": user_id,
                "item_id": f"{user_id}-i{idx}",
                "memory_type": "fact",
                "t_elapsed": float(idx + 1),
                "review_index": idx + 1,
                "outcome": 1,
                "grade": None,
                "metadata": {},
            })

    split = split_review_events(events, seed=42)

    train_users = {e["user_id"] for e in split["train"]}
    valid_users = {e["user_id"] for e in split["valid"]}
    test_users = {e["user_id"] for e in split["test"]}

    assert train_users.isdisjoint(valid_users)
    assert train_users.isdisjoint(test_users)
    assert valid_users.isdisjoint(test_users)


def test_split_review_events_preserves_user_event_order():
    events = [
        {"user_id": "u1", "item_id": "i1", "memory_type": "fact", "t_elapsed": 1.0, "review_index": 2, "outcome": 1, "grade": None, "metadata": {}},
        {"user_id": "u1", "item_id": "i1", "memory_type": "fact", "t_elapsed": 1.0, "review_index": 1, "outcome": 1, "grade": None, "metadata": {}},
    ]

    split = split_review_events(events, seed=1, train_ratio=1.0, valid_ratio=0.0)

    assert [e["review_index"] for e in split["train"]] == [1, 2]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_split.py -v`
Expected: FAIL with `ImportError: cannot import name 'split_review_events'`

**Step 3: Write minimal implementation**

Add to `src/memory_decay/human_data.py`:

```python
import random
from collections import defaultdict


def split_review_events(
    events: list[dict],
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        grouped[event["user_id"]].append(event)
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["review_index"], row["t_elapsed"], row["item_id"]))

    users = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(users)

    n_train = int(len(users) * train_ratio)
    n_valid = int(len(users) * valid_ratio)
    train_users = set(users[:n_train])
    valid_users = set(users[n_train:n_train + n_valid])
    test_users = set(users[n_train + n_valid:])

    return {
        "train": [event for user in users for event in grouped[user] if user in train_users],
        "valid": [event for user in users for event in grouped[user] if user in valid_users],
        "test": [event for user in users for event in grouped[user] if user in test_users],
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_split.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
cd /Users/lit/memory-decay
git add src/memory_decay/human_data.py tests/test_human_split.py
git commit -m "Add leakage-safe human review split helper"
```

---

### Task 4: Add parameter search for fact-side calibration

**Files:**
- Create: `src/memory_decay/human_optimizer.py`
- Test: `tests/test_human_optimizer.py`

**Step 1: Write the failing test**

Create `tests/test_human_optimizer.py`:

```python
from memory_decay.human_optimizer import random_search_human_params


BASE_EVENTS = [
    {"user_id": "u1", "item_id": "i1", "memory_type": "fact", "t_elapsed": 1.0, "review_index": 1, "outcome": 1, "grade": None, "metadata": {}},
    {"user_id": "u1", "item_id": "i1", "memory_type": "fact", "t_elapsed": 10.0, "review_index": 2, "outcome": 0, "grade": None, "metadata": {}},
    {"user_id": "u2", "item_id": "i2", "memory_type": "fact", "t_elapsed": 2.0, "review_index": 1, "outcome": 1, "grade": None, "metadata": {}},
]


def test_random_search_returns_best_params_and_metrics():
    result = random_search_human_params(
        train_events=BASE_EVENTS,
        valid_events=BASE_EVENTS,
        iterations=4,
        seed=42,
    )

    assert set(result) >= {"best_params", "best_metrics", "trials"}
    assert len(result["trials"]) == 4
    assert "lambda_fact" in result["best_params"]
    assert "nll" in result["best_metrics"]


def test_random_search_keeps_episode_params_fixed():
    result = random_search_human_params(
        train_events=BASE_EVENTS,
        valid_events=BASE_EVENTS,
        iterations=2,
        seed=1,
    )

    assert result["best_params"]["lambda_episode"] == 0.035
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_optimizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory_decay.human_optimizer'`

**Step 3: Write minimal implementation**

Create `src/memory_decay/human_optimizer.py`:

```python
from __future__ import annotations

import random

from .human_eval import HumanCalibrationEvaluator


DEFAULT_PARAMS = {
    "lambda_fact": 0.02,
    "lambda_episode": 0.035,
    "beta_fact": 0.08,
    "beta_episode": 0.12,
    "alpha": 0.5,
    "stability_weight": 0.8,
    "stability_decay": 0.01,
    "reinforcement_gain_direct": 0.2,
    "reinforcement_gain_assoc": 0.05,
    "stability_cap": 1.0,
}


def sample_fact_params(rng: random.Random) -> dict:
    params = dict(DEFAULT_PARAMS)
    params["lambda_fact"] = rng.uniform(0.005, 0.08)
    params["stability_weight"] = rng.uniform(0.0, 2.0)
    params["stability_decay"] = rng.uniform(0.0, 0.1)
    params["reinforcement_gain_direct"] = rng.uniform(0.05, 0.5)
    return params


def random_search_human_params(
    *,
    train_events: list[dict],
    valid_events: list[dict],
    iterations: int = 25,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    trials = []
    best_params = None
    best_metrics = None
    best_score = float("inf")

    for _ in range(iterations):
        params = sample_fact_params(rng)
        observation_params = {"activation_scale": 6.0, "bias": -3.0, "stability_scale": 0.0}
        train_eval = HumanCalibrationEvaluator(params, observation_params)
        train_eval.evaluate(train_events)
        valid_eval = HumanCalibrationEvaluator(params, observation_params)
        metrics = valid_eval.evaluate(valid_events)
        score = metrics["nll"]
        trials.append({"params": params, "metrics": metrics})
        if score < best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    return {"best_params": best_params, "best_metrics": best_metrics, "trials": trials}
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_optimizer.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
cd /Users/lit/memory-decay
git add src/memory_decay/human_optimizer.py tests/test_human_optimizer.py
git commit -m "Add random-search human calibration optimizer"
```

---

### Task 5: Add CLI runner that produces calibration artifacts

**Files:**
- Create: `src/memory_decay/human_runner.py`
- Create: `tests/test_human_runner.py`

**Step 1: Write the failing test**

Create `tests/test_human_runner.py`:

```python
import json

from memory_decay.human_runner import run_human_calibration


def test_run_human_calibration_writes_result_files(tmp_path):
    events_path = tmp_path / "events.jsonl"
    rows = [
        {"user_id": "u1", "item_id": "i1", "memory_type": "fact", "t_elapsed": 1.0, "review_index": 1, "outcome": 1, "grade": None, "metadata": {}},
        {"user_id": "u1", "item_id": "i1", "memory_type": "fact", "t_elapsed": 8.0, "review_index": 2, "outcome": 0, "grade": None, "metadata": {}},
        {"user_id": "u2", "item_id": "i2", "memory_type": "fact", "t_elapsed": 2.0, "review_index": 1, "outcome": 1, "grade": None, "metadata": {}},
    ]
    events_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    output_dir = tmp_path / "out"
    result = run_human_calibration(str(events_path), str(output_dir), iterations=3, seed=42)

    assert (output_dir / "best_params.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "trials.json").exists()
    assert "best_params" in result
    assert "test_metrics" in result
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory_decay.human_runner'`

**Step 3: Write minimal implementation**

Create `src/memory_decay/human_runner.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from .human_data import load_review_events_jsonl, split_review_events
from .human_eval import HumanCalibrationEvaluator
from .human_optimizer import random_search_human_params


def run_human_calibration(
    events_path: str,
    output_dir: str,
    *,
    iterations: int = 25,
    seed: int = 42,
) -> dict:
    events = load_review_events_jsonl(events_path)
    split = split_review_events(events, seed=seed)
    result = random_search_human_params(
        train_events=split["train"],
        valid_events=split["valid"] or split["train"],
        iterations=iterations,
        seed=seed,
    )
    evaluator = HumanCalibrationEvaluator(
        result["best_params"],
        {"activation_scale": 6.0, "bias": -3.0, "stability_scale": 0.0},
    )
    test_metrics = evaluator.evaluate(split["test"] or split["train"])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_params.json").write_text(json.dumps(result["best_params"], indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (out / "trials.json").write_text(json.dumps(result["trials"], indent=2), encoding="utf-8")

    return {
        "best_params": result["best_params"],
        "valid_metrics": result["best_metrics"],
        "test_metrics": test_metrics,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_runner.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
cd /Users/lit/memory-decay
git add src/memory_decay/human_runner.py tests/test_human_runner.py
git commit -m "Add human calibration runner and artifacts"
```

---

### Task 6: Bridge fitted params into the existing synthetic benchmark

**Files:**
- Modify: `src/memory_decay/main.py`
- Create: `tests/test_human_to_synthetic_bridge.py`

**Step 1: Write the failing test**

Create `tests/test_human_to_synthetic_bridge.py`:

```python
import json

from memory_decay.main import merge_human_calibrated_params


def test_merge_human_calibrated_params_overrides_fact_side_only(tmp_path):
    path = tmp_path / "best_params.json"
    path.write_text(json.dumps({
        "lambda_fact": 0.011,
        "stability_weight": 1.2,
        "stability_decay": 0.02,
        "reinforcement_gain_direct": 0.31,
        "lambda_episode": 999.0
    }), encoding="utf-8")

    params = {
        "lambda_fact": 0.05,
        "lambda_episode": 0.08,
        "beta_fact": 0.3,
        "beta_episode": 0.5,
        "alpha": 0.5,
        "stability_weight": 0.8,
        "stability_decay": 0.01,
        "reinforcement_gain_direct": 0.2,
        "reinforcement_gain_assoc": 0.05,
        "stability_cap": 1.0,
    }

    merged = merge_human_calibrated_params(params, str(path))

    assert merged["lambda_fact"] == 0.011
    assert merged["stability_weight"] == 1.2
    assert merged["stability_decay"] == 0.02
    assert merged["reinforcement_gain_direct"] == 0.31
    assert merged["lambda_episode"] == 0.08
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_to_synthetic_bridge.py -v`
Expected: FAIL with `ImportError: cannot import name 'merge_human_calibrated_params'`

**Step 3: Write minimal implementation**

Add to `src/memory_decay/main.py`:

```python
def merge_human_calibrated_params(base_params: dict, best_params_path: str) -> dict:
    with open(best_params_path, "r", encoding="utf-8") as f:
        fitted = json.load(f)

    merged = dict(base_params)
    for key in (
        "lambda_fact",
        "stability_weight",
        "stability_decay",
        "reinforcement_gain_direct",
    ):
        if key in fitted:
            merged[key] = fitted[key]
    return merged
```

Then extend `run_experiment(...)` with an optional argument:

```python
def run_experiment(..., calibrated_params_path: Optional[str] = None, ...):
```

Apply it after base params are constructed:

```python
if calibrated_params_path:
    params = merge_human_calibrated_params(params, calibrated_params_path)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_human_to_synthetic_bridge.py -v`
Expected: 1 passed

**Step 5: Run focused regression tests**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_simulation.py tests/test_runner.py tests/test_human_data.py tests/test_human_eval.py tests/test_human_optimizer.py tests/test_human_runner.py tests/test_human_to_synthetic_bridge.py -q`
Expected: all selected tests pass

**Step 6: Commit**

```bash
cd /Users/lit/memory-decay
git add src/memory_decay/main.py tests/test_human_to_synthetic_bridge.py
git commit -m "Bridge human-calibrated fact params into synthetic benchmark"
```

---

### Task 7: Document the new workflow and scientific boundaries

**Files:**
- Modify: `README.md`
- Modify: `docs/research-log.md`

**Step 1: Write the failing documentation checklist**

Add this checklist to your working notes and treat it as the failing doc spec:

```text
- README explains the new human calibration pipeline separately from synthetic experiments
- README shows one command for human calibration and one command for synthetic replay with fitted params
- research-log states that human data calibrates only fact-side parameters
- research-log states that episode behavior remains constrained extrapolation, not directly fit
```

**Step 2: Update README with exact commands**

Add a new section:

```markdown
## Human Calibration Workflow

1. Normalize a real review log into JSONL with fields:
   `user_id`, `item_id`, `memory_type`, `t_elapsed`, `review_index`, `outcome`, `grade`, `metadata`
2. Fit fact-side parameters:

```bash
cd /Users/lit/memory-decay
PYTHONPATH=src uv run python -m memory_decay.human_runner data/human_reviews.jsonl outputs/human_calibration
```

3. Reuse fitted params in the synthetic benchmark:

```bash
cd /Users/lit/memory-decay
PYTHONPATH=src uv run python -m memory_decay.main --dataset data/memories_500.jsonl --calibrated-params outputs/human_calibration/best_params.json
```
```

**Step 3: Update research log with scientific scope limits**

Add to `docs/research-log.md`:

```markdown
## Human Calibration Scope

- Real review logs are used only to fit `fact`-side forgetting and reinforcement parameters.
- `episode` parameters are not learned from these datasets because open large-scale episodic-memory review logs are not available.
- Synthetic graph retrieval remains the external validation environment for association behavior and fact/episode contrast.
```

**Step 4: Run sanity checks**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest -q`
Expected: full suite passes

**Step 5: Commit**

```bash
cd /Users/lit/memory-decay
git add README.md docs/research-log.md
git commit -m "Document human calibration workflow and research boundaries"
```

---

### Task 8: Run the first end-to-end scientific smoke test

**Files:**
- Use existing: `data/test_50.jsonl`
- Use existing: `data/memories_50.jsonl`
- Output: `outputs/human_calibration_smoke/`

**Step 1: Create a tiny synthetic human-events fixture**

Create `data/human_reviews_smoke.jsonl` with a few fact-only rows:

```json
{"user_id":"u1","item_id":"f1","memory_type":"fact","t_elapsed":1.0,"review_index":1,"outcome":1,"grade":null,"metadata":{}}
{"user_id":"u1","item_id":"f1","memory_type":"fact","t_elapsed":9.0,"review_index":2,"outcome":0,"grade":null,"metadata":{}}
{"user_id":"u2","item_id":"f2","memory_type":"fact","t_elapsed":2.0,"review_index":1,"outcome":1,"grade":null,"metadata":{}}
```

**Step 2: Run human calibration**

Run:

```bash
cd /Users/lit/memory-decay
PYTHONPATH=src uv run python -m memory_decay.human_runner data/human_reviews_smoke.jsonl outputs/human_calibration_smoke
```

Expected:
- `outputs/human_calibration_smoke/best_params.json` exists
- `outputs/human_calibration_smoke/metrics.json` exists

**Step 3: Run synthetic replay with calibrated params**

Run:

```bash
cd /Users/lit/memory-decay
PYTHONPATH=src uv run python -m memory_decay.main --dataset data/memories_50.jsonl --calibrated-params outputs/human_calibration_smoke/best_params.json --improvement-budget 0
```

Expected:
- experiment completes without API dependency
- reported recall/precision/overall metrics are produced

**Step 4: Write the result note**

Record in `docs/research-log.md`:

```markdown
## Human Calibration Smoke Test

- Date: 2026-03-18
- Human fixture: `data/human_reviews_smoke.jsonl`
- Output dir: `outputs/human_calibration_smoke/`
- Synthetic replay dataset: `data/memories_50.jsonl`
- Result: calibration artifacts generated and synthetic benchmark ran with fitted fact parameters
```

**Step 5: Commit**

```bash
cd /Users/lit/memory-decay
git add data/human_reviews_smoke.jsonl docs/research-log.md
git commit -m "Add end-to-end human calibration smoke test"
```
