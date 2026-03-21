"""Loader for dashboard-visible output artifacts under outputs/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OutputRecord = dict[str, Any]


def _safe_load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, "File not found"
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"
    if not isinstance(data, dict):
        return None, "Top-level JSON value must be an object"
    return data, None


def _parse_error_record(record_type: str, source_file: Path, error: str) -> OutputRecord:
    return {
        "record_type": record_type,
        "record_id": f"parse_error:{source_file}",
        "dataset_name": None,
        "suite_name": None,
        "source_dir": str(source_file.parent),
        "source_file": str(source_file),
        "status": "parse_error",
        "error": error,
    }


def _incomplete_record(
    record_type: str,
    source_file: Path,
    error: str,
    *,
    dataset_name: str | None = None,
    suite_name: str | None = None,
    source_dir: Path | None = None,
) -> OutputRecord:
    return {
        "record_type": record_type,
        "record_id": f"incomplete:{source_file}",
        "dataset_name": dataset_name,
        "suite_name": suite_name,
        "source_dir": str(source_dir or source_file.parent),
        "source_file": str(source_file),
        "status": "incomplete",
        "error": error,
    }


def _build_benchmark_record(
    run: dict[str, Any],
    source_file: Path,
    dataset_name: str,
    suite_name: str | None,
    run_dir: Path,
) -> OutputRecord:
    required = ("inputs", "artifacts", "baseline", "calibrated", "delta")
    missing = [key for key in required if key not in run]
    if missing:
        return _incomplete_record(
            "benchmark_run",
            source_file,
            f"Missing required keys: {', '.join(missing)}",
            dataset_name=dataset_name,
            suite_name=suite_name,
            source_dir=run_dir,
        )

    inputs = run.get("inputs") or {}
    artifacts = run.get("artifacts") or {}
    calibration = run.get("human_calibration") or {}
    valid = calibration.get("valid_metrics") or {}
    test = calibration.get("test_metrics") or {}
    baseline = run.get("baseline") or {}
    calibrated = run.get("calibrated") or {}
    delta = run.get("delta") or {}

    return {
        "record_type": "benchmark_run",
        "record_id": f"{suite_name or 'standalone'}:{dataset_name}:{run_dir}",
        "dataset_name": dataset_name,
        "suite_name": suite_name,
        "source_dir": str(run_dir),
        "source_file": str(source_file),
        "status": "completed",
        "error": None,
        "baseline_overall": baseline.get("overall_score"),
        "baseline_retrieval": baseline.get("retrieval_score"),
        "baseline_plausibility": baseline.get("plausibility_score"),
        "calibrated_overall": calibrated.get("overall_score"),
        "calibrated_retrieval": calibrated.get("retrieval_score"),
        "calibrated_plausibility": calibrated.get("plausibility_score"),
        "delta_overall": delta.get("overall_score"),
        "delta_retrieval": delta.get("retrieval_score"),
        "delta_plausibility": delta.get("plausibility_score"),
        "calibration_valid_nll": valid.get("nll"),
        "calibration_valid_brier": valid.get("brier"),
        "calibration_valid_ece": valid.get("ece"),
        "calibration_test_nll": test.get("nll"),
        "calibration_test_brier": test.get("brier"),
        "calibration_test_ece": test.get("ece"),
        "num_iterations": inputs.get("iterations"),
        "seed": inputs.get("seed"),
        "total_ticks": inputs.get("total_ticks"),
        "eval_interval": inputs.get("eval_interval"),
        "decay_type": inputs.get("decay_type"),
        "reactivation_policy": inputs.get("reactivation_policy"),
        "embedding_backend": inputs.get("embedding_backend"),
        "baseline_output_path": artifacts.get("baseline_output"),
        "calibrated_output_path": artifacts.get("calibrated_output"),
        "best_params_path": artifacts.get("best_params_path"),
        "raw_inputs": inputs,
    }


def _build_calibration_record(metrics: dict[str, Any], source_file: Path) -> OutputRecord:
    required = ("nll", "brier", "ece", "num_events")
    missing = [key for key in required if key not in metrics]
    if missing:
        return _incomplete_record(
            "human_calibration",
            source_file,
            f"Missing required keys: {', '.join(missing)}",
            source_dir=source_file.parent,
        )

    return {
        "record_type": "human_calibration",
        "record_id": f"calibration:{source_file.parent}",
        "dataset_name": None,
        "suite_name": None,
        "source_dir": str(source_file.parent),
        "source_file": str(source_file),
        "status": "completed",
        "error": None,
        "nll": metrics.get("nll"),
        "brier": metrics.get("brier"),
        "ece": metrics.get("ece"),
        "num_events": metrics.get("num_events"),
        "best_params_path": str(source_file.parent / "best_params.json")
        if (source_file.parent / "best_params.json").exists()
        else None,
        "trials_path": str(source_file.parent / "trials.json")
        if (source_file.parent / "trials.json").exists()
        else None,
    }


def load_output_records(outputs_dir: str) -> list[OutputRecord]:
    """Load dashboard-visible output artifacts from outputs/."""
    root = Path(outputs_dir)
    if not root.exists():
        return []

    records: list[OutputRecord] = []
    suite_run_dirs: set[Path] = set()
    calibration_dirs: set[Path] = set()

    for suite_path in sorted(root.rglob("suite_summary.json")):
        suite, error = _safe_load_json(suite_path)
        if error is not None:
            records.append(_parse_error_record("benchmark_run", suite_path, error))
            continue

        runs = suite.get("runs")
        if not isinstance(runs, dict):
            records.append(
                _incomplete_record(
                    "benchmark_run",
                    suite_path,
                    "Missing required keys: runs",
                    source_dir=suite_path.parent,
                )
            )
            continue

        for dataset_name, run in runs.items():
            if not isinstance(run, dict):
                records.append(
                    _incomplete_record(
                        "benchmark_run",
                        suite_path,
                        f"Run entry for {dataset_name} must be an object",
                        dataset_name=dataset_name,
                        suite_name=suite_path.parent.name,
                        source_dir=suite_path.parent / dataset_name,
                    )
                )
                continue

            run_dir = (suite_path.parent / dataset_name).resolve()
            suite_run_dirs.add(run_dir)
            records.append(
                _build_benchmark_record(
                    run,
                    suite_path,
                    dataset_name,
                    suite_path.parent.name,
                    run_dir,
                )
            )

    for comparison_path in sorted(root.rglob("comparison_summary.json")):
        if comparison_path.parent.resolve() in suite_run_dirs:
            continue
        summary, error = _safe_load_json(comparison_path)
        if error is not None:
            records.append(_parse_error_record("benchmark_run", comparison_path, error))
            continue
        records.append(
            _build_benchmark_record(
                summary,
                comparison_path,
                comparison_path.parent.name,
                None,
                comparison_path.parent.resolve(),
            )
        )

    for metrics_path in sorted(root.rglob("metrics.json")):
        if metrics_path.parent.name != "human_calibration":
            continue
        calib_dir = metrics_path.parent.resolve()
        if calib_dir in calibration_dirs:
            continue
        calibration_dirs.add(calib_dir)

        metrics, error = _safe_load_json(metrics_path)
        if error is not None:
            records.append(_parse_error_record("human_calibration", metrics_path, error))
            continue
        records.append(_build_calibration_record(metrics, metrics_path))

    return records
