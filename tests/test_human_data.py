"""Tests for human review data normalization helpers."""

import json

from memory_decay.human_data import (
    load_review_events_jsonl,
    normalize_anki_event,
    normalize_duolingo_event,
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
    assert event["metadata"]["history_correct"] == 3
    assert event["metadata"]["history_seen"] == 5


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
        {"user_id": "u1", "item_id": "i2"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    events = load_review_events_jsonl(path)

    assert len(events) == 1
    assert events[0]["item_id"] == "i1"
