"""Orchestrate human calibration and synthetic baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .human_runner import run_human_calibration
from .main import run_experiment


def _extract_scores(result: dict) -> dict:
    summary = result.get("initial_score_summary", {})
    return {
        "overall_score": float(result.get("initial_overall_score", 0.0)),
        "retrieval_score": float(summary.get("retrieval_score", 0.0)),
        "plausibility_score": float(summary.get("plausibility_score", 0.0)),
    }


def run_calibration_benchmark(
    *,
    human_events_path: str,
    synthetic_dataset_path: str,
    output_dir: str,
    iterations: int = 25,
    seed: int = 42,
    total_ticks: int = 100,
    eval_interval: int = 5,
    decay_type: str = "exponential",
    reactivation_policy: str = "none",
    embedding_backend: str = "auto",
) -> dict:
    """Run human calibration, then compare baseline vs calibrated synthetic runs."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    calibration_dir = output_root / "human_calibration"
    calibration_result = run_human_calibration(
        human_events_path,
        str(calibration_dir),
        iterations=iterations,
        seed=seed,
    )
    best_params_path = calibration_dir / "best_params.json"

    baseline_output = output_root / "baseline.json"
    calibrated_output = output_root / "calibrated.json"

    baseline_result = run_experiment(
        decay_type=decay_type,
        total_ticks=total_ticks,
        eval_interval=eval_interval,
        reactivation_policy=reactivation_policy,
        embedding_backend=embedding_backend,
        improvement_budget=0,
        dataset_path=synthetic_dataset_path,
        output_path=str(baseline_output),
        seed=seed,
    )
    calibrated_result = run_experiment(
        decay_type=decay_type,
        total_ticks=total_ticks,
        eval_interval=eval_interval,
        reactivation_policy=reactivation_policy,
        embedding_backend=embedding_backend,
        improvement_budget=0,
        dataset_path=synthetic_dataset_path,
        calibrated_params_path=str(best_params_path),
        output_path=str(calibrated_output),
        seed=seed,
    )

    baseline_scores = _extract_scores(baseline_result)
    calibrated_scores = _extract_scores(calibrated_result)
    delta = {
        key: round(calibrated_scores[key] - baseline_scores[key], 4)
        for key in baseline_scores
    }

    summary = {
        "inputs": {
            "human_events_path": human_events_path,
            "synthetic_dataset_path": synthetic_dataset_path,
            "iterations": iterations,
            "seed": seed,
            "total_ticks": total_ticks,
            "eval_interval": eval_interval,
            "decay_type": decay_type,
            "reactivation_policy": reactivation_policy,
            "embedding_backend": embedding_backend,
        },
        "artifacts": {
            "calibration_dir": str(calibration_dir),
            "best_params_path": str(best_params_path),
            "baseline_output": str(baseline_output),
            "calibrated_output": str(calibrated_output),
        },
        "human_calibration": calibration_result,
        "baseline": baseline_scores,
        "calibrated": calibrated_scores,
        "delta": delta,
    }

    summary_path = output_root / "comparison_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run_calibration_benchmark_suite(
    *,
    human_events_path: str,
    synthetic_dataset_paths: list[str],
    output_dir: str,
    iterations: int = 25,
    seed: int = 42,
    total_ticks: int = 100,
    eval_interval: int = 5,
    decay_type: str = "exponential",
    reactivation_policy: str = "none",
    embedding_backend: str = "auto",
) -> dict:
    """Run the calibration benchmark across multiple synthetic datasets."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    runs = {}
    for dataset_path in synthetic_dataset_paths:
        dataset_name = Path(dataset_path).stem
        run_output_dir = output_root / dataset_name
        runs[dataset_name] = run_calibration_benchmark(
            human_events_path=human_events_path,
            synthetic_dataset_path=dataset_path,
            output_dir=str(run_output_dir),
            iterations=iterations,
            seed=seed,
            total_ticks=total_ticks,
            eval_interval=eval_interval,
            decay_type=decay_type,
            reactivation_policy=reactivation_policy,
            embedding_backend=embedding_backend,
        )

    suite_summary = {
        "inputs": {
            "human_events_path": human_events_path,
            "synthetic_dataset_paths": synthetic_dataset_paths,
            "iterations": iterations,
            "seed": seed,
            "total_ticks": total_ticks,
            "eval_interval": eval_interval,
            "decay_type": decay_type,
            "reactivation_policy": reactivation_policy,
            "embedding_backend": embedding_backend,
        },
        "runs": runs,
    }
    (output_root / "suite_summary.json").write_text(
        json.dumps(suite_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return suite_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run human calibration and compare baseline vs calibrated synthetic runs"
    )
    parser.add_argument(
        "--human-events",
        default="data/human_reviews_smoke.jsonl",
        help="Path to normalized human review JSONL",
    )
    parser.add_argument(
        "--synthetic-dataset",
        action="append",
        dest="synthetic_datasets",
        help="Path to a synthetic benchmark dataset. Repeat to run multiple datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pre_program_pipeline",
        help="Directory to write calibration and comparison artifacts",
    )
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-ticks", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument(
        "--decay-type",
        choices=["exponential", "power_law"],
        default="exponential",
    )
    parser.add_argument(
        "--reactivation-policy",
        choices=["none", "random", "scheduled_query"],
        default="none",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "local", "gemini"],
        default="auto",
    )
    args = parser.parse_args()
    synthetic_datasets = args.synthetic_datasets or [
        "data/memories_50.jsonl",
        "data/memories_500.jsonl",
    ]

    suite = run_calibration_benchmark_suite(
        human_events_path=args.human_events,
        synthetic_dataset_paths=synthetic_datasets,
        output_dir=args.output_dir,
        iterations=args.iterations,
        seed=args.seed,
        total_ticks=args.total_ticks,
        eval_interval=args.eval_interval,
        decay_type=args.decay_type,
        reactivation_policy=args.reactivation_policy,
        embedding_backend=args.embedding_backend,
    )

    parts = []
    for dataset_name, summary in suite["runs"].items():
        parts.append(
            f"{dataset_name}: "
            f"{summary['baseline']['overall_score']:.4f}"
            f"->{summary['calibrated']['overall_score']:.4f} "
            f"({summary['delta']['overall_score']:+.4f})"
        )
    print("Done: " + " | ".join(parts))


if __name__ == "__main__":
    main()
