"""Tests for human review event splitting."""

from memory_decay.human_data import split_review_events


def test_split_review_events_keeps_users_disjoint():
    events = []
    for user_id in ["u1", "u2", "u3", "u4", "u5"]:
        for idx in range(3):
            events.append(
                {
                    "user_id": user_id,
                    "item_id": f"{user_id}-i{idx}",
                    "memory_type": "fact",
                    "t_elapsed": float(idx + 1),
                    "review_index": idx + 1,
                    "outcome": 1,
                    "grade": None,
                    "metadata": {},
                }
            )

    split = split_review_events(events, seed=42)

    train_users = {e["user_id"] for e in split["train"]}
    valid_users = {e["user_id"] for e in split["valid"]}
    test_users = {e["user_id"] for e in split["test"]}

    assert train_users.isdisjoint(valid_users)
    assert train_users.isdisjoint(test_users)
    assert valid_users.isdisjoint(test_users)


def test_split_review_events_preserves_user_event_order():
    events = [
        {
            "user_id": "u1",
            "item_id": "i1",
            "memory_type": "fact",
            "t_elapsed": 1.0,
            "review_index": 2,
            "outcome": 1,
            "grade": None,
            "metadata": {},
        },
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
    ]

    split = split_review_events(events, seed=1, train_ratio=1.0, valid_ratio=0.0)

    assert [e["review_index"] for e in split["train"]] == [1, 2]
