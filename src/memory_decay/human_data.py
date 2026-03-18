"""Helpers for normalizing and splitting human review logs."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REQUIRED_EVENT_KEYS = {
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
    """Normalize a Duolingo-style review row into the common event schema."""
    return {
        "user_id": str(raw["user_id"]),
        "item_id": str(raw["lexeme_id"]),
        "memory_type": "fact",
        "t_elapsed": float(raw["delta"]),
        "review_index": int(raw["history_seen"]),
        "outcome": 1 if float(raw["p_recall"]) >= 0.5 else 0,
        "grade": None,
        "metadata": {
            "history_correct": int(raw.get("history_correct", 0)),
            "history_seen": int(raw.get("history_seen", 0)),
        },
    }


def normalize_anki_event(raw: dict) -> dict:
    """Normalize an Anki-style review row into the common event schema."""
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
    """Load only rows that already match the common human review schema."""
    events: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if REQUIRED_EVENT_KEYS.issubset(row):
                events.append(row)
    return events


def split_review_events(
    events: list[dict],
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> dict[str, list[dict]]:
    """Split by user so no user appears in multiple partitions."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        grouped[str(event["user_id"])].append(event)

    for user_events in grouped.values():
        user_events.sort(
            key=lambda row: (row["review_index"], row["t_elapsed"], row["item_id"])
        )

    users = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(users)

    if not users:
        return {"train": [], "valid": [], "test": []}

    n_train = int(len(users) * train_ratio)
    n_valid = int(len(users) * valid_ratio)

    if n_train <= 0:
        n_train = 1
    if n_train + n_valid >= len(users):
        n_valid = max(0, len(users) - n_train - 1)

    train_users = set(users[:n_train])
    valid_users = set(users[n_train : n_train + n_valid])
    test_users = set(users[n_train + n_valid :])

    return {
        "train": [
            event for user in users if user in train_users for event in grouped[user]
        ],
        "valid": [
            event for user in users if user in valid_users for event in grouped[user]
        ],
        "test": [event for user in users if user in test_users for event in grouped[user]],
    }
