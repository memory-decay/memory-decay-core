"""CLI runner for human review calibration."""

from __future__ import annotations

import argparse
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
    """Fit fact-side parameters on human review events and save artifacts."""
    events = load_review_events_jsonl(events_path)
    split = split_review_events(events, seed=seed)

    valid_events = split["valid"] or split["train"]
    test_events = split["test"] or split["train"]

    result = random_search_human_params(
        train_events=split["train"],
        valid_events=valid_events,
        iterations=iterations,
        seed=seed,
    )

    evaluator = HumanCalibrationEvaluator(
        result["best_params"],
        {"activation_scale": 6.0, "bias": -3.0, "stability_scale": 0.0},
    )
    test_metrics = evaluator.evaluate(test_events)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "best_params.json").write_text(
        json.dumps(result["best_params"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "trials.json").write_text(
        json.dumps(result["trials"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "best_params": result["best_params"],
        "valid_metrics": result["best_metrics"],
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run human review calibration")
    parser.add_argument("events_path", help="Path to normalized human review JSONL")
    parser.add_argument("output_dir", help="Directory to write calibration artifacts")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_human_calibration(
        args.events_path,
        args.output_dir,
        iterations=args.iterations,
        seed=args.seed,
    )
    print(
        "Done: "
        f"nll={result['test_metrics']['nll']:.4f} "
        f"brier={result['test_metrics']['brier']:.4f} "
        f"ece={result['test_metrics']['ece']:.4f}"
    )


if __name__ == "__main__":
    main()
