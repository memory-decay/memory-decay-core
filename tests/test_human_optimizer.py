"""Tests for human calibration parameter search."""

from memory_decay.human_optimizer import random_search_human_params


BASE_EVENTS = [
    {
        "user_id": "u1",
        "item_id": "i1",
        "memory_type": "fact",
        "t_elapsed": 1.0,
        "review_index": 1,
        "outcome": 1,
        "grade": None,
        "metadata": {},
    },
    {
        "user_id": "u1",
        "item_id": "i1",
        "memory_type": "fact",
        "t_elapsed": 10.0,
        "review_index": 2,
        "outcome": 0,
        "grade": None,
        "metadata": {},
    },
    {
        "user_id": "u2",
        "item_id": "i2",
        "memory_type": "fact",
        "t_elapsed": 2.0,
        "review_index": 1,
        "outcome": 1,
        "grade": None,
        "metadata": {},
    },
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
