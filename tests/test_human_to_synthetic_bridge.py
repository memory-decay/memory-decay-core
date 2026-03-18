"""Tests for applying calibrated human params to the synthetic benchmark."""

import json

from memory_decay.main import merge_human_calibrated_params


def test_merge_human_calibrated_params_overrides_fact_side_only(tmp_path):
    path = tmp_path / "best_params.json"
    path.write_text(
        json.dumps(
            {
                "lambda_fact": 0.011,
                "stability_weight": 1.2,
                "stability_decay": 0.02,
                "reinforcement_gain_direct": 0.31,
                "lambda_episode": 999.0,
            }
        ),
        encoding="utf-8",
    )

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
