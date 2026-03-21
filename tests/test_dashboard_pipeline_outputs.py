from __future__ import annotations

import json
import sys
from pathlib import Path

from dashboard.output_loader import load_output_records


def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _find_components(layout, predicate):
    results = []

    def _walk(node):
        if hasattr(node, "id") and predicate(node):
            results.append(node)
        children = getattr(node, "children", None)
        if isinstance(children, list):
            for child in children:
                _walk(child)
        elif hasattr(children, "id"):
            _walk(children)

    _walk(layout)
    return results


def test_load_suite_summary_creates_benchmark_rows(tmp_path: Path):
    _write_json(
        tmp_path / "pre_program_pipeline" / "suite_summary.json",
        {
            "runs": {
                "memories_50": {
                    "inputs": {
                        "iterations": 25,
                        "seed": 42,
                        "total_ticks": 100,
                        "eval_interval": 5,
                        "decay_type": "exponential",
                        "reactivation_policy": "none",
                        "embedding_backend": "local",
                    },
                    "artifacts": {
                        "baseline_output": "outputs/pre_program_pipeline/memories_50/baseline.json",
                        "calibrated_output": "outputs/pre_program_pipeline/memories_50/calibrated.json",
                        "best_params_path": "outputs/pre_program_pipeline/memories_50/human_calibration/best_params.json",
                    },
                    "human_calibration": {
                        "valid_metrics": {"nll": 0.1, "brier": 0.2, "ece": 0.3},
                        "test_metrics": {"nll": 0.4, "brier": 0.5, "ece": 0.6},
                    },
                    "baseline": {
                        "overall_score": 0.1,
                        "retrieval_score": 0.2,
                        "plausibility_score": 0.3,
                    },
                    "calibrated": {
                        "overall_score": 0.4,
                        "retrieval_score": 0.5,
                        "plausibility_score": 0.6,
                    },
                    "delta": {
                        "overall_score": 0.3,
                        "retrieval_score": 0.3,
                        "plausibility_score": 0.3,
                    },
                }
            }
        },
    )

    records = load_output_records(str(tmp_path))

    benchmark_rows = [r for r in records if r["record_type"] == "benchmark_run"]
    assert len(benchmark_rows) == 1
    assert benchmark_rows[0]["dataset_name"] == "memories_50"
    assert benchmark_rows[0]["delta_overall"] == 0.3


def test_suite_and_nested_comparison_are_deduped(tmp_path: Path):
    suite_dir = tmp_path / "pre_program_pipeline"
    run_dir = suite_dir / "memories_50"
    _write_json(
        suite_dir / "suite_summary.json",
        {
            "runs": {
                "memories_50": {
                    "inputs": {
                        "iterations": 25,
                        "seed": 42,
                        "total_ticks": 100,
                        "eval_interval": 5,
                        "decay_type": "exponential",
                        "reactivation_policy": "none",
                        "embedding_backend": "local",
                    },
                    "artifacts": {},
                    "human_calibration": {},
                    "baseline": {
                        "overall_score": 0.1,
                        "retrieval_score": 0.2,
                        "plausibility_score": 0.3,
                    },
                    "calibrated": {
                        "overall_score": 0.2,
                        "retrieval_score": 0.3,
                        "plausibility_score": 0.4,
                    },
                    "delta": {
                        "overall_score": 0.1,
                        "retrieval_score": 0.1,
                        "plausibility_score": 0.1,
                    },
                }
            }
        },
    )
    _write_json(
        run_dir / "comparison_summary.json",
        {
            "inputs": {
                "iterations": 25,
                "seed": 42,
                "total_ticks": 100,
                "eval_interval": 5,
                "decay_type": "exponential",
                "reactivation_policy": "none",
                "embedding_backend": "local",
            },
            "artifacts": {},
            "human_calibration": {},
            "baseline": {
                "overall_score": 0.1,
                "retrieval_score": 0.2,
                "plausibility_score": 0.3,
            },
            "calibrated": {
                "overall_score": 0.2,
                "retrieval_score": 0.3,
                "plausibility_score": 0.4,
            },
            "delta": {
                "overall_score": 0.1,
                "retrieval_score": 0.1,
                "plausibility_score": 0.1,
            },
        },
    )

    records = load_output_records(str(tmp_path))
    benchmark_rows = [r for r in records if r["record_type"] == "benchmark_run"]
    assert len(benchmark_rows) == 1


def test_standalone_human_calibration_creates_calibration_row(tmp_path: Path):
    calib_dir = tmp_path / "human_calibration"
    _write_json(
        calib_dir / "metrics.json",
        {"nll": 1.0, "brier": 0.2, "ece": 0.3, "num_events": 5},
    )
    _write_json(calib_dir / "best_params.json", {"lambda_fact": 0.01})
    _write_json(calib_dir / "trials.json", [])

    records = load_output_records(str(tmp_path))
    calibration_rows = [r for r in records if r["record_type"] == "human_calibration"]
    assert len(calibration_rows) == 1
    assert calibration_rows[0]["nll"] == 1.0
    assert calibration_rows[0]["status"] == "completed"


def test_malformed_output_json_surfaces_parse_error(tmp_path: Path):
    bad_file = tmp_path / "human_calibration" / "metrics.json"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("{not-json", encoding="utf-8")

    records = load_output_records(str(tmp_path))
    assert len(records) == 1
    assert records[0]["status"] == "parse_error"


def test_missing_required_summary_keys_marks_incomplete(tmp_path: Path):
    _write_json(
        tmp_path / "run" / "comparison_summary.json",
        {"baseline": {"overall_score": 0.1}},
    )

    records = load_output_records(str(tmp_path))
    assert len(records) == 1
    assert records[0]["status"] == "incomplete"


def test_missing_optional_artifacts_do_not_fail_completed_row(tmp_path: Path):
    _write_json(
        tmp_path / "run" / "comparison_summary.json",
        {
            "inputs": {
                "iterations": 10,
                "seed": 1,
                "total_ticks": 20,
                "eval_interval": 5,
                "decay_type": "exponential",
                "reactivation_policy": "none",
                "embedding_backend": "local",
            },
            "artifacts": {},
            "human_calibration": {},
            "baseline": {
                "overall_score": 0.1,
                "retrieval_score": 0.2,
                "plausibility_score": 0.3,
            },
            "calibrated": {
                "overall_score": 0.2,
                "retrieval_score": 0.3,
                "plausibility_score": 0.4,
            },
            "delta": {
                "overall_score": 0.1,
                "retrieval_score": 0.1,
                "plausibility_score": 0.1,
            },
        },
    )

    records = load_output_records(str(tmp_path))
    assert len(records) == 1
    assert records[0]["status"] == "completed"
    assert records[0]["best_params_path"] is None


def test_app_layout_has_pipeline_tab_and_view():
    sys.modules.pop("dashboard.app", None)
    import dashboard.app as app_module

    layout = app_module.app.layout
    assert _find_components(layout, lambda n: getattr(n, "id", None) == "tab-pipeline")
    assert _find_components(layout, lambda n: getattr(n, "id", None) == "pipeline-view")


def test_filter_pipeline_records_by_type_and_search():
    sys.modules.pop("dashboard.app", None)
    import dashboard.app as app_module

    records = [
        {
            "record_type": "benchmark_run",
            "dataset_name": "memories_50",
            "suite_name": "pre_program_pipeline",
            "source_dir": "outputs/a",
            "status": "completed",
        },
        {
            "record_type": "human_calibration",
            "dataset_name": None,
            "suite_name": None,
            "source_dir": "outputs/human_calibration",
            "status": "completed",
        },
    ]

    filtered = app_module._filter_pipeline_records(records, "benchmark_run", "memories_50")
    assert len(filtered) == 1
    assert filtered[0]["record_type"] == "benchmark_run"


def test_build_pipeline_detail_view_for_benchmark():
    sys.modules.pop("dashboard.app", None)
    import dashboard.app as app_module

    detail = app_module._build_pipeline_detail_view(
        {
            "record_type": "benchmark_run",
            "record_id": "suite:memories_50",
            "dataset_name": "memories_50",
            "suite_name": "pre_program_pipeline",
            "status": "completed",
            "baseline_overall": 0.1,
            "calibrated_overall": 0.4,
            "delta_overall": 0.3,
            "baseline_output_path": "outputs/pre_program_pipeline/memories_50/baseline.json",
            "calibrated_output_path": "outputs/pre_program_pipeline/memories_50/calibrated.json",
            "best_params_path": "outputs/pre_program_pipeline/memories_50/human_calibration/best_params.json",
        }
    )
    assert "memories_50" in str(detail)
    assert "0.3000" in str(detail)


def test_build_pipeline_detail_view_for_calibration():
    sys.modules.pop("dashboard.app", None)
    import dashboard.app as app_module

    detail = app_module._build_pipeline_detail_view(
        {
            "record_type": "human_calibration",
            "record_id": "calib:outputs/human_calibration",
            "source_dir": "outputs/human_calibration",
            "status": "completed",
            "nll": 1.0,
            "brier": 0.2,
            "ece": 0.3,
            "num_events": 5,
            "best_params_path": "outputs/human_calibration/best_params.json",
            "trials_path": "outputs/human_calibration/trials.json",
            "source_file": "outputs/human_calibration/metrics.json",
        }
    )
    assert "1.0000" in str(detail)
    assert "5" in str(detail)


def test_update_page_display_supports_pipeline_page():
    sys.modules.pop("dashboard.app", None)
    import dashboard.app as app_module

    result = app_module.update_page_display("pipeline", None)
    pipeline_view_style = result[8]
    active_tab_style = result[-1]
    assert pipeline_view_style["display"] == "block"
    assert active_tab_style["borderBottom"] == "2px solid #1565C0"


def test_pipeline_count_text_reflects_filtered_rows():
    sys.modules.pop("dashboard.app", None)
    import dashboard.app as app_module

    records = [
        {
            "record_type": "benchmark_run",
            "dataset_name": "memories_50",
            "suite_name": "suite_a",
            "source_dir": "outputs/a",
            "status": "completed",
        },
        {
            "record_type": "human_calibration",
            "dataset_name": None,
            "suite_name": None,
            "source_dir": "outputs/b",
            "status": "completed",
        },
    ]

    row_data, count_children = app_module._build_pipeline_table_state(records, "All", "")
    assert len(row_data) == 2
    assert "Showing 2 pipeline records" in str(count_children)


def test_empty_pipeline_state_message():
    sys.modules.pop("dashboard.app", None)
    import dashboard.app as app_module

    row_data, count_children = app_module._build_pipeline_table_state([], "All", "")
    assert row_data == []
    assert "Showing 0 pipeline records" in str(count_children)
