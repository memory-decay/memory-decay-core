"""Tests for the human calibration runner."""

import json

from memory_decay.human_runner import run_human_calibration


def test_run_human_calibration_writes_result_files(tmp_path):
    events_path = tmp_path / "events.jsonl"
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
        {
            "user_id": "u1",
            "item_id": "i1",
            "memory_type": "fact",
            "t_elapsed": 8.0,
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
    events_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    result = run_human_calibration(
        str(events_path),
        str(output_dir),
        iterations=3,
        seed=42,
    )

    assert (output_dir / "best_params.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "trials.json").exists()
    assert "best_params" in result
    assert "test_metrics" in result
