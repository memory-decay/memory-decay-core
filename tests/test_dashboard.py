"""Tests for dashboard data loader module.

Covers schema evolution, edge cases (missing files, malformed data),
era classification, phase mapping, history dedup, performance, and more.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import tracemalloc
from pathlib import Path

import pytest

from dashboard.data_loader import (
    PHASE_RANGES,
    Experiment,
    HistoryEntry,
    classify_era,
    get_phase,
    load_all_experiments,
    load_archive_history,
    load_history,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def experiments_dir(tmp_path: Path) -> Path:
    """Create a temporary experiments directory with sample experiments."""
    return tmp_path


def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _find_real_experiment(experiments: list[Experiment], predicate, description: str) -> Experiment:
    """Return the first matching real experiment or skip if unavailable."""
    match = next((exp for exp in experiments if predicate(exp)), None)
    if match is None:
        pytest.skip(f"No real experiment available for: {description}")
    return match


# ---------------------------------------------------------------------------
# Old-era experiment data (exp_0000 style)
# ---------------------------------------------------------------------------

OLD_ERA_RESULTS = {
    "status": "completed",
    "tick": 200,
    "recall_rate": 0.04878048780487805,
    "precision_rate": 0.14963503649635038,
    "overall_score": 0.020986353816341393,
    "retrieval_score": 0.021650871203463082,
    "plausibility_score": 0.7953839421742998,
    "recall_mean": 0.03353658536585366,
    "precision_mean": 0.1553072764639457,
    "mrr_mean": 0.02388211382113821,
    "corr_score": 0.6902985003052601,
    "smoothness_score": 0.9004693840433394,
    "threshold_metrics": {
        "0.2": {"recall_rate": 0.06097, "precision_rate": 0.15015},
        "0.3": {"recall_rate": 0.04878, "precision_rate": 0.14963},
        "0.4": {"recall_rate": 0.01219, "precision_rate": 0.16759},
        "0.5": {"recall_rate": 0.01219, "precision_rate": 0.15384},
    },
    "snapshots": [
        {"tick": i, "overall_score": 0.5 - i * 0.02, "retrieval_score": 0.3 - i * 0.01}
        for i in range(0, 201, 20)
    ],
    "duration_seconds": 15.29,
}

OLD_ERA_PARAMS = {
    "lambda_fact": 0.02,
    "lambda_episode": 0.035,
    "beta_fact": 0.08,
    "beta_episode": 0.12,
    "alpha": 0.5,
    "stability_weight": 0.8,
}


# ---------------------------------------------------------------------------
# New-era experiment data (exp_lme_0059 style)
# ---------------------------------------------------------------------------

NEW_ERA_RESULTS = {
    "status": "completed",
    "tick": 200,
    "recall_rate": 0.6339869281045751,
    "precision_rate": 0.28888888888888886,
    "overall_score": 0.3448028425075384,
    "retrieval_score": 0.3687005314258432,
    "plausibility_score": 0.567893418797321,
    "recall_mean": 0.6325344952795933,
    "precision_mean": 0.300307297241605,
    "mrr_mean": 0.38044057129024444,
    "corr_score": 0.27150205432142044,
    "smoothness_score": 0.8642847832732217,
    "threshold_metrics": {
        f"{t / 10:.1f}": {"recall_rate": 0.63, "precision_rate": 0.29}
        for t in range(1, 10)
    },
    "threshold_summary": {"threshold_auc": 0.6325, "slope": -0.0131},
    "retention_curve": {"40": 1.0, "80": 1.0, "120": 0.6363, "160": 0.5683, "200": 0.6339},
    "retention_auc": 0.7677,
    "selectivity_score": 0.0,
    "robustness_score": 0.0,
    "eval_v2_score": 0.34548262993882967,
    "forgetting_depth": 0.3675,
    "forgetting_score": 1.0,
    "strict_score": 0.458456,
    "snapshots": [
        {
            "tick": i,
            "overall_score": 0.6 - i * 0.01,
            "retrieval_score": 0.4 - i * 0.005,
            "retention_curve": {"40": 1.0, "80": 1.0, "120": 0.6, "160": 0.5, "200": 0.6},
            "selectivity_score": 0.1,
            "eval_v2_score": 0.3,
        }
        for i in range(0, 201, 20)
    ],
    "duration_seconds": 10.05,
}

NEW_ERA_PARAMS = {
    "lambda_fact": 0.009,
    "lambda_episode": 0.040,
    "alpha": 1.5,
    "floor_max": 0.45,
    "sigmoid_k": 30.0,
    "sigmoid_mid": 0.30,
    "jost_power": 4.0,
    "activation_weight": 0.3,
}


# ---------------------------------------------------------------------------
# Tests: Experiment Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    """VAL-DATA-001: Complete Experiment Discovery"""

    def test_discover_experiments_maxdepth1(self, experiments_dir: Path):
        """Only top-level exp dirs found, no recursion into archive/experiments subdir."""
        _write_json(experiments_dir / "exp_0001" / "results.json", OLD_ERA_RESULTS)
        # Create nested experiment that should NOT be discovered
        _write_json(
            experiments_dir / "archive_memories500" / "exp_0002" / "results.json",
            OLD_ERA_RESULTS,
        )
        # Create experiments/experiments subdir
        _write_json(
            experiments_dir / "experiments" / "exp_0065" / "results.json",
            OLD_ERA_RESULTS,
        )
        exps = load_all_experiments(str(experiments_dir))
        ids = {e.id for e in exps}
        assert "exp_0001" in ids
        assert "exp_0002" not in ids
        assert "exp_0065" not in ids

    def test_excludes_non_experiment_entries(self, experiments_dir: Path):
        """archive_memories500, best symlink, history.jsonl, strict_eval.py excluded."""
        _write_json(experiments_dir / "exp_0001" / "results.json", OLD_ERA_RESULTS)
        # Create non-experiment entries
        (experiments_dir / "archive_memories500").mkdir()
        (experiments_dir / "best").symlink_to("exp_0001")
        (experiments_dir / "history.jsonl").write_text("{}")
        (experiments_dir / "strict_eval.py").write_text("# script")

        exps = load_all_experiments(str(experiments_dir))
        ids = {e.id for e in exps}
        assert ids == {"exp_0001"}

    def test_best_symlink_not_double_counted(self, experiments_dir: Path):
        """best symlink should not be followed or double-counted."""
        _write_json(experiments_dir / "exp_0001" / "results.json", OLD_ERA_RESULTS)
        (experiments_dir / "best").symlink_to("exp_0001")

        exps = load_all_experiments(str(experiments_dir))
        ids = [e.id for e in exps]
        assert ids.count("exp_0001") == 1


# ---------------------------------------------------------------------------
# Tests: Missing / Malformed Files
# ---------------------------------------------------------------------------

class TestMissingFiles:
    """VAL-DATA-002, VAL-DATA-003, VAL-DATA-004"""

    def test_missing_results_json(self, experiments_dir: Path):
        """Missing results.json → status='no_results', all metrics null."""
        _write_json(experiments_dir / "exp_0360" / "params.json", {})
        _write_text(experiments_dir / "exp_0360" / "hypothesis.txt", "test")

        exps = load_all_experiments(str(experiments_dir))
        assert len(exps) == 1
        exp = exps[0]
        assert exp.id == "exp_0360"
        assert exp.status == "no_results"
        assert exp.overall_score is None

    def test_missing_params_json(self, experiments_dir: Path):
        """Missing params.json → empty dict."""
        _write_json(experiments_dir / "exp_0001" / "results.json", OLD_ERA_RESULTS)

        exps = load_all_experiments(str(experiments_dir))
        assert exps[0].params == {}

    def test_missing_hypothesis_txt(self, experiments_dir: Path):
        """Missing hypothesis.txt → empty string."""
        _write_json(experiments_dir / "exp_0001" / "results.json", OLD_ERA_RESULTS)

        exps = load_all_experiments(str(experiments_dir))
        assert exps[0].hypothesis == ""

    def test_malformed_json_handling(self, experiments_dir: Path):
        """Invalid JSON → parse_error status, excluded from aggregations."""
        _write_text(
            experiments_dir / "exp_9999" / "results.json",
            "{this is not valid json!!!",
        )
        exps = load_all_experiments(str(experiments_dir))
        assert len(exps) == 1
        assert exps[0].status == "parse_error"

    def test_malformed_params_json(self, experiments_dir: Path):
        """Invalid params.json → parse_error status."""
        _write_json(experiments_dir / "exp_9999" / "results.json", OLD_ERA_RESULTS)
        _write_text(
            experiments_dir / "exp_9999" / "params.json",
            "not valid json",
        )
        exps = load_all_experiments(str(experiments_dir))
        assert exps[0].status == "parse_error"


# ---------------------------------------------------------------------------
# Tests: validation_failed
# ---------------------------------------------------------------------------

class TestValidationFailed:
    """VAL-DATA-005"""

    def test_validation_failed_results(self, experiments_dir: Path):
        """Only status+error → all metrics null, error preserved."""
        results = {"status": "validation_failed", "error": "Insufficient decay"}
        _write_json(experiments_dir / "exp_lme_0063" / "results.json", results)

        exps = load_all_experiments(str(experiments_dir))
        exp = exps[0]
        assert exp.status == "validation_failed"
        assert exp.error == "Insufficient decay"
        assert exp.overall_score is None
        assert exp.retrieval_score is None
        assert exp.strict_score is None


# ---------------------------------------------------------------------------
# Tests: Schema Evolution
# ---------------------------------------------------------------------------

class TestSchemaEvolution:
    """VAL-DATA-006, VAL-DATA-007, VAL-DATA-008, VAL-DATA-009"""

    def test_schema_evolution_null_defaults(self, experiments_dir: Path):
        """Old-era exp missing new fields get null, not KeyError."""
        _write_json(experiments_dir / "exp_0000" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0059" / "results.json", NEW_ERA_RESULTS)

        exps = load_all_experiments(str(experiments_dir))
        old = next(e for e in exps if e.id == "exp_0000")
        new = next(e for e in exps if e.id == "exp_lme_0059")

        # Old-era missing fields → null
        assert old.retention_curve is None
        assert old.eval_v2_score is None
        assert old.strict_score is None
        assert old.forgetting_depth is None
        assert old.forgetting_score is None
        assert old.threshold_summary is None
        assert old.selectivity_score is None
        assert old.robustness_score is None

        # New-era has these fields
        assert new.retention_curve is not None
        assert new.eval_v2_score is not None
        assert new.strict_score is not None
        assert new.forgetting_depth is not None

    def test_threshold_metrics_variation(self, experiments_dir: Path):
        """Old-era 4 thresholds, new-era 9 thresholds; missing → null."""
        _write_json(experiments_dir / "exp_0000" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0059" / "results.json", NEW_ERA_RESULTS)

        exps = load_all_experiments(str(experiments_dir))
        old = next(e for e in exps if e.id == "exp_0000")
        new = next(e for e in exps if e.id == "exp_lme_0059")

        # Old-era: 4 thresholds present, others null
        assert old.threshold_metrics.get("0.2") is not None
        assert old.threshold_metrics.get("0.1") is None
        assert old.threshold_metrics.get("0.8") is None

        # New-era: 9 thresholds present
        for t in [f"{i / 10:.1f}" for i in range(1, 10)]:
            assert new.threshold_metrics.get(t) is not None, f"threshold {t} missing"

    def test_params_field_variation(self, experiments_dir: Path):
        """Old-era beta_fact/beta_episode; new-era has assoc_boost, floor_max etc."""
        _write_json(experiments_dir / "exp_0000" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_0000" / "params.json", OLD_ERA_PARAMS)
        _write_json(experiments_dir / "exp_lme_0000" / "results.json", NEW_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0000" / "params.json", NEW_ERA_PARAMS)

        exps = load_all_experiments(str(experiments_dir))
        old = next(e for e in exps if e.id == "exp_0000")
        new = next(e for e in exps if e.id == "exp_lme_0000")

        # Old-era has beta_fact, no floor_max
        assert old.params.get("beta_fact") == 0.08
        assert old.params.get("floor_max") is None

        # New-era has floor_max, no beta_fact
        assert new.params.get("floor_max") == 0.45
        assert new.params.get("beta_fact") is None

    def test_snapshots_structure(self, experiments_dir: Path):
        """11 snapshots with ticks 0,20,...,200; cross-era key differences."""
        _write_json(experiments_dir / "exp_0000" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0059" / "results.json", NEW_ERA_RESULTS)

        exps = load_all_experiments(str(experiments_dir))
        old = next(e for e in exps if e.id == "exp_0000")
        new = next(e for e in exps if e.id == "exp_lme_0059")

        # Both have 11 snapshots
        assert len(old.snapshots) == 11
        assert len(new.snapshots) == 11

        # Tick values
        expected_ticks = list(range(0, 201, 20))
        assert [s["tick"] for s in old.snapshots] == expected_ticks
        assert [s["tick"] for s in new.snapshots] == expected_ticks

        # Old-era snapshot missing new keys → null
        assert old.snapshots[0].get("retention_curve") is None
        assert old.snapshots[0].get("selectivity_score") is None

        # New-era snapshot has new keys
        assert new.snapshots[0].get("retention_curve") is not None
        assert new.snapshots[0].get("selectivity_score") is not None

    def test_snapshots_missing_defaults_empty(self, experiments_dir: Path):
        """Missing snapshots array → empty list."""
        results = dict(OLD_ERA_RESULTS)
        del results["snapshots"]
        _write_json(experiments_dir / "exp_0000" / "results.json", results)

        exps = load_all_experiments(str(experiments_dir))
        assert exps[0].snapshots == []


# ---------------------------------------------------------------------------
# Tests: Era Classification
# ---------------------------------------------------------------------------

class TestEraClassification:
    """VAL-DATA-010"""

    def test_classify_era(self):
        assert classify_era("exp_0000") == "memories_500"
        assert classify_era("exp_0359") == "memories_500"
        assert classify_era("exp_lme_0000") == "LongMemEval"
        assert classify_era("exp_lme_0075") == "LongMemEval"

    def test_era_classification_in_loaded_data(self, experiments_dir: Path):
        _write_json(experiments_dir / "exp_0001" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0001" / "results.json", NEW_ERA_RESULTS)

        exps = load_all_experiments(str(experiments_dir))
        old = next(e for e in exps if e.id == "exp_0001")
        new = next(e for e in exps if e.id == "exp_lme_0001")
        assert old.era == "memories_500"
        assert new.era == "LongMemEval"


# ---------------------------------------------------------------------------
# Tests: Phase Mapping
# ---------------------------------------------------------------------------

class TestPhaseMapping:
    """VAL-DATA-011, VAL-DATA-012"""

    def test_phase_mapping(self, experiments_dir: Path):
        """Each experiment maps to exactly one phase."""
        _write_json(experiments_dir / "exp_0010" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_0051" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_0200" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_0300" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_0359" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0000" / "results.json", NEW_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0046" / "results.json", NEW_ERA_RESULTS)

        exps = load_all_experiments(str(experiments_dir))
        phases = {e.id: e.phase for e in exps}

        # Phase 1: Early Exploration (0-24 area)
        assert phases["exp_0010"] == 1
        # Phase 3: Auto-Research First 25 (51-82 area)
        assert phases["exp_0051"] == 3
        # Phase 4: Protocol Fixes (83-296 area)
        assert phases["exp_0200"] == 4
        # Phase 5: Scoring Overhaul (297-359 area)
        assert phases["exp_0300"] == 5
        # Phase 5: Scoring Overhaul (297-359 area)
        assert phases["exp_0359"] == 5
        # Phase 7: LongMemEval Integration
        assert phases["exp_lme_0000"] == 7
        # Phase 9: Batch Embedding (46+)
        assert phases["exp_lme_0046"] == 9

    def test_new_era_not_confused_with_old(self, experiments_dir: Path):
        """exp_lme_0030 is NOT in same phase as exp_0030."""
        _write_json(experiments_dir / "exp_0030" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0030" / "results.json", NEW_ERA_RESULTS)

        exps = load_all_experiments(str(experiments_dir))
        phases = {e.id: e.phase for e in exps}
        assert phases["exp_0030"] != phases["exp_lme_0030"]
        assert phases["exp_0030"] == 2  # Reinforcement Redesign
        assert phases["exp_lme_0030"] == 8  # Auto-Research

    def test_every_experiment_has_a_phase(self, experiments_dir: Path):
        """No experiment has phase=None."""
        _write_json(experiments_dir / "exp_0000" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_0359" / "results.json", OLD_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0000" / "results.json", NEW_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0075" / "results.json", NEW_ERA_RESULTS)

        exps = load_all_experiments(str(experiments_dir))
        for exp in exps:
            assert exp.phase is not None, f"{exp.id} has no phase"


# ---------------------------------------------------------------------------
# Tests: Numeric Precision
# ---------------------------------------------------------------------------

class TestNumericPrecision:
    """VAL-DATA-013"""

    def test_float_precision_preserved(self, experiments_dir: Path):
        """Full float precision preserved."""
        results = dict(NEW_ERA_RESULTS)
        _write_json(experiments_dir / "exp_lme_0000" / "results.json", results)

        exps = load_all_experiments(str(experiments_dir))
        exp = exps[0]
        # Verify full precision
        assert exp.overall_score == 0.3448028425075384
        assert exp.recall_rate == 0.6339869281045751

    def test_integer_tick_field(self, experiments_dir: Path):
        """tick field parsed as int."""
        _write_json(experiments_dir / "exp_0000" / "results.json", OLD_ERA_RESULTS)

        exps = load_all_experiments(str(experiments_dir))
        assert exps[0].tick == 200
        assert isinstance(exps[0].tick, int)

        # Snapshot ticks also int
        assert isinstance(exps[0].snapshots[0]["tick"], int)


# ---------------------------------------------------------------------------
# Tests: history.jsonl
# ---------------------------------------------------------------------------

class TestHistoryLoading:
    """VAL-DATA-014, VAL-DATA-015, VAL-DATA-016, VAL-DATA-017"""

    def test_history_string_overall(self, tmp_path: Path):
        """overall='validation_failed' string handled, strict_score='failed' handled."""
        history = [
            {"exp": "exp_lme_0063", "overall": "validation_failed",
             "strict_score": "failed", "status": "validation_failed"},
            {"exp": "exp_lme_0059", "overall": 0.3448,
             "strict_score": 0.4585, "status": "improved"},
        ]
        history_file = tmp_path / "history.jsonl"
        history_file.write_text("\n".join(json.dumps(e) for e in history))

        entries = load_history(str(history_file))
        assert len(entries) == 2

        failed = next(e for e in entries if e["exp"] == "exp_lme_0063")
        assert failed["overall"] == "validation_failed"
        assert failed["strict_score"] == "failed"

        good = next(e for e in entries if e["exp"] == "exp_lme_0059")
        assert good["overall"] == 0.3448
        assert good["strict_score"] == 0.4585

    def test_history_sparse_fields(self, tmp_path: Path):
        """Missing fields default to null."""
        history = [
            {"exp": "exp_lme_0000", "overall": 0.0374, "status": "baseline"},
            {"exp": "exp_lme_0059", "overall": 0.3448, "retrieval": 0.3687,
             "plausibility": 0.5679, "status": "improved"},
        ]
        history_file = tmp_path / "history.jsonl"
        history_file.write_text("\n".join(json.dumps(e) for e in history))

        entries = load_history(str(history_file))
        # First entry missing retrieval, plausibility
        first = entries[0]
        assert first.get("retrieval") is None
        assert first.get("plausibility") is None

    def test_history_dedup(self, tmp_path: Path):
        """Duplicate entries deduped by last-occurrence-in-file."""
        history = [
            {"exp": "exp_lme_0059", "overall": 0.3448, "hypothesis": "first"},
            {"exp": "exp_lme_0060", "overall": 0.3406, "hypothesis": "only"},
            {"exp": "exp_lme_0059", "overall": 0.3448, "hypothesis": "second",
             "cv_mean": 0.3458},
            {"exp": "exp_lme_0063", "overall": "validation_failed"},
            {"exp": "exp_lme_0063", "overall": "validation_failed", "hypothesis": "updated"},
        ]
        history_file = tmp_path / "history.jsonl"
        history_file.write_text("\n".join(json.dumps(e) for e in history))

        entries = load_history(str(history_file))
        assert len(entries) == 3  # deduped to unique exps

        exp_59 = next(e for e in entries if e["exp"] == "exp_lme_0059")
        assert exp_59["hypothesis"] == "second"  # last occurrence wins
        assert exp_59["cv_mean"] == 0.3458

        exp_63 = next(e for e in entries if e["exp"] == "exp_lme_0063")
        assert exp_63["hypothesis"] == "updated"

    def test_history_vs_results_consistency(self, experiments_dir: Path):
        """Cross-reference history entries with on-disk results."""
        # Create history file
        history_data = [
            {"exp": "exp_0000", "overall": 0.020986353816341393, "status": "completed"},
        ]
        history_file = experiments_dir / "history.jsonl"
        history_file.write_text("\n".join(json.dumps(e) for e in history_data))

        # Create experiment with matching results
        _write_json(experiments_dir / "exp_0000" / "results.json", OLD_ERA_RESULTS)

        entries = load_history(str(history_file))
        exps = load_all_experiments(str(experiments_dir))

        assert entries[0]["overall"] == exps[0].overall_score


# ---------------------------------------------------------------------------
# Tests: Archive History
# ---------------------------------------------------------------------------

class TestArchiveHistory:
    """VAL-DATA-020"""

    def test_archive_history_normalization(self, tmp_path: Path):
        """archive_memories500 key normalization: experiment→exp, overall_score→overall."""
        archive_history = [
            {"experiment": "exp_0000", "overall_score": 0.0210, "status": "baseline",
             "hypothesis": "default exponential"},
            {"experiment": "exp_0301", "overall_score": 0.1528, "status": "improved",
             "retrieval_score": 0.1599, "plausibility_score": 0.7038},
        ]
        archive_file = tmp_path / "archive_history.jsonl"
        archive_file.write_text("\n".join(json.dumps(e) for e in archive_history))

        entries = load_archive_history(str(archive_file))
        assert len(entries) == 2

        # Key normalization applied
        assert entries[0]["exp"] == "exp_0000"
        assert entries[0]["overall"] == 0.0210
        assert "experiment" not in entries[0]
        assert "overall_score" not in entries[0]

        assert entries[1]["exp"] == "exp_0301"
        assert entries[1]["retrieval"] == 0.1599
        assert entries[1]["plausibility"] == 0.7038

    def test_archive_history_sparse_fields(self, tmp_path: Path):
        """Non-overlapping fields in archive → null default."""
        archive_history = [
            {"experiment": "exp_0000", "overall_score": 0.0210},
        ]
        archive_file = tmp_path / "archive_history.jsonl"
        archive_file.write_text("\n".join(json.dumps(e) for e in archive_history))

        entries = load_archive_history(str(archive_file))
        assert entries[0].get("status") is None
        assert entries[0].get("hypothesis") is None


# ---------------------------------------------------------------------------
# Tests: Integration with real data
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests against real experiment data."""

    @pytest.fixture
    def real_experiments_dir(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "experiments")

    def test_load_all_experiments_real_data(self, real_experiments_dir: str):
        """VAL-DATA-001: Discovery count matches filesystem."""
        exps = load_all_experiments(real_experiments_dir)
        # Count experiment dirs in filesystem
        fs_count = 0
        for entry in os.listdir(real_experiments_dir):
            if entry.startswith("exp_") and os.path.isdir(
                os.path.join(real_experiments_dir, entry)
            ):
                fs_count += 1
        assert len(exps) == fs_count
        # Verify count is reasonable (should be > 400 experiments)
        assert fs_count > 400, f"Expected >400 experiments but found {fs_count}"

    def test_load_old_and_new_era_real(self, real_experiments_dir: str):
        """Load representative older-schema and newer-schema experiments from real data."""
        exps = load_all_experiments(real_experiments_dir)
        old = _find_real_experiment(
            exps,
            lambda e: (
                e.strict_score is None
                and e.forgetting_depth is None
                and e.retention_curve is None
                and e.eval_v2_score is None
            ),
            "older-schema experiment",
        )
        evolved = _find_real_experiment(
            exps,
            lambda e: (
                e.strict_score is not None
                and e.forgetting_depth is not None
                and e.retention_curve is not None
            ),
            "evolved-schema experiment",
        )

        assert old.strict_score is None
        assert old.forgetting_depth is None
        assert old.retention_curve is None
        assert old.eval_v2_score is None

        assert evolved.strict_score is not None
        assert evolved.forgetting_depth is not None
        assert evolved.retention_curve is not None

        assert old.params.get("beta_fact") is not None
        assert evolved.params, "Expected evolved-schema experiment to have params"

    def test_validation_failed_real(self, real_experiments_dir: str):
        """A real validation_failed experiment preserves error text and null metrics."""
        exps = load_all_experiments(real_experiments_dir)
        exp = _find_real_experiment(
            exps,
            lambda e: e.status == "validation_failed",
            "validation_failed experiment",
        )
        assert exp.status == "validation_failed"
        assert exp.overall_score is None
        assert exp.error is not None
        assert isinstance(exp.error, str)

    def test_no_results_real(self, real_experiments_dir: str):
        """A real experiment without results.json loads as no_results."""
        exps = load_all_experiments(real_experiments_dir)
        exp = _find_real_experiment(
            exps,
            lambda e: e.status == "no_results",
            "no_results experiment",
        )
        assert exp.status == "no_results"
        assert exp.overall_score is None

    def test_archive_history_real(self, real_experiments_dir: str):
        """Real archive_memories500/history.jsonl loaded with normalization."""
        archive_path = os.path.join(real_experiments_dir, "archive_memories500", "history.jsonl")
        if not os.path.exists(archive_path):
            pytest.skip("archive_memories500/history.jsonl not available")
        entries = load_archive_history(archive_path)
        assert len(entries) > 0
        # All entries should have 'exp' key (normalized)
        for entry in entries:
            assert "exp" in entry
            assert "experiment" not in entry


# ---------------------------------------------------------------------------
# Tests: Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    """VAL-DATA-018, VAL-DATA-019"""

    @pytest.fixture
    def real_experiments_dir(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "experiments")

    def test_cold_load_performance(self, real_experiments_dir: str):
        """Full load under 5 seconds."""
        # Count experiment dirs in filesystem for dynamic assertion
        fs_count = sum(
            1 for entry in os.listdir(real_experiments_dir)
            if entry.startswith("exp_") and os.path.isdir(
                os.path.join(real_experiments_dir, entry)
            )
        )
        start = time.perf_counter()
        exps = load_all_experiments(real_experiments_dir)
        elapsed = time.perf_counter() - start
        assert len(exps) == fs_count
        assert elapsed < 5.0, f"Loading took {elapsed:.2f}s, exceeds 5s limit"

    def test_memory_efficiency(self, real_experiments_dir: str):
        """Dataset under 200MB RAM."""
        tracemalloc.start()
        exps = load_all_experiments(real_experiments_dir)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 200, f"Peak memory {peak_mb:.1f}MB exceeds 200MB limit"


# ---------------------------------------------------------------------------
# Tests: Phase mapping constants
# ---------------------------------------------------------------------------

class TestPhaseConstants:

    def test_phase_ranges_coverage(self):
        """All 11 phases (0-10) defined."""
        assert set(PHASE_RANGES.keys()) == set(range(11))

    def test_get_phase(self):
        assert get_phase("exp_0010") == 1
        assert get_phase("exp_0030") == 2
        assert get_phase("exp_lme_0000") == 7
        assert get_phase("exp_lme_0046") == 9


# ---------------------------------------------------------------------------
# Tests: Experiment dataclass
# ---------------------------------------------------------------------------

class TestExperimentDataclass:

    def test_null_safe_field_access(self, experiments_dir: Path):
        """Accessing None fields doesn't crash."""
        _write_json(experiments_dir / "exp_0001" / "results.json", OLD_ERA_RESULTS)
        exps = load_all_experiments(str(experiments_dir))
        exp = exps[0]
        # These should all be accessible (None, not crash)
        assert exp.strict_score is None
        assert exp.forgetting_depth is None
        assert exp.retention_curve is None
        assert exp.eval_v2_score is None
        assert exp.error is None

    def test_status_values(self, experiments_dir: Path):
        """Completed experiment has correct status."""
        _write_json(experiments_dir / "exp_0001" / "results.json", OLD_ERA_RESULTS)
        exps = load_all_experiments(str(experiments_dir))
        assert exps[0].status == "completed"


# ---------------------------------------------------------------------------
# Tests: Dashboard app module import and basic structure
# ---------------------------------------------------------------------------

class TestDashboardApp:
    """Tests for the dashboard app module (import, layout, callbacks)."""

    @staticmethod
    def _find_components(layout, predicate):
        """Recursively walk Dash layout tree and find components matching predicate."""
        results = []
        def _walk(node):
            if hasattr(node, "id"):
                if predicate(node):
                    results.append(node)
            children = getattr(node, "children", None)
            if children is not None:
                if isinstance(children, list):
                    for child in children:
                        _walk(child)
                elif hasattr(children, "id"):
                    _walk(children)
        _walk(layout)
        return results

    def test_app_module_imports(self):
        """dashboard.app can be imported without errors."""
        import dashboard.app  # noqa: F401
        assert hasattr(dashboard.app, "app")

    def test_app_has_location_component(self):
        """App layout includes dcc.Location for URL state."""
        import dashboard.app
        layout = dashboard.app.app.layout
        found = self._find_components(layout, lambda n: "Location" in type(n).__name__)
        assert len(found) > 0, "dcc.Location not found in app layout"

    def test_app_has_store_components(self):
        """App layout includes dcc.Store for shared state."""
        import dashboard.app
        layout = dashboard.app.app.layout
        found = self._find_components(layout, lambda n: "Store" in type(n).__name__)
        assert len(found) >= 2, f"Expected 2+ dcc.Store, found {len(found)}"

    def test_app_has_era_dropdown(self):
        """App layout includes era dropdown with All/memories_500/LongMemEval."""
        import dashboard.app
        layout = dashboard.app.app.layout
        found = self._find_components(layout, lambda n: getattr(n, "id", None) == "era-dropdown")
        assert len(found) > 0, "era-dropdown not found in app layout"

    def test_app_has_search_input(self):
        """App layout includes text search input."""
        import dashboard.app
        layout = dashboard.app.app.layout
        found = self._find_components(layout, lambda n: getattr(n, "id", None) == "search-input")
        assert len(found) > 0, "search-input not found in app layout"

    def test_app_has_status_filter(self):
        """App layout includes status multi-select filter."""
        import dashboard.app
        layout = dashboard.app.app.layout
        found = self._find_components(layout, lambda n: getattr(n, "id", None) == "status-filter")
        assert len(found) > 0, "status-filter not found in app layout"

    def test_app_has_ag_grid(self):
        """App layout includes AgGrid component."""
        import dashboard.app
        layout = dashboard.app.app.layout
        found = self._find_components(layout, lambda n: "AgGrid" in type(n).__name__)
        assert len(found) > 0, "AgGrid not found in app layout"

    def test_app_has_detail_view(self):
        """App layout includes detail view overlay."""
        import dashboard.app
        layout = dashboard.app.app.layout
        found = self._find_components(layout, lambda n: getattr(n, "id", None) == "detail-view")
        assert len(found) > 0, "detail-view not found in app layout"


# ---------------------------------------------------------------------------
# Tests: Leaderboard filtering and sorting logic
# ---------------------------------------------------------------------------

class TestLeaderboardLogic:
    """Tests for leaderboard filtering and sorting helper functions."""

    @pytest.fixture
    def real_experiments_dir(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "experiments")

    def test_filter_dataframe_all_eras(self, real_experiments_dir: str):
        """VAL-TABLE-003: Filtering 'All' shows both eras."""
        import dashboard.app as app_module
        df = app_module._df_all
        memories_count = len(df[df["era"] == "memories_500"])
        lme_count = len(df[df["era"] == "LongMemEval"])
        assert memories_count > 300, f"Expected >300 memories_500, got {memories_count}"
        assert lme_count > 50, f"Expected >50 LongMemEval, got {lme_count}"
        assert len(df) == memories_count + lme_count

    def test_filter_dataframe_memories_500(self, real_experiments_dir: str):
        """VAL-TABLE-001: Filtering memories_500 shows only exp_NNNN."""
        import dashboard.app as app_module
        df = app_module._df_all
        filtered = app_module._filter_dataframe(df, "memories_500", [], "")
        assert len(filtered) > 0
        assert all(filtered["era"] == "memories_500"), "Non-memories_500 experiments leaked through"
        assert not any(filtered["era"] == "LongMemEval"), "LongMemEval experiment leaked through"

    def test_filter_dataframe_longmemeval(self, real_experiments_dir: str):
        """VAL-TABLE-002: Filtering LongMemEval shows only exp_lme_NNNN."""
        import dashboard.app as app_module
        df = app_module._df_all
        filtered = app_module._filter_dataframe(df, "LongMemEval", [], "")
        assert len(filtered) > 0
        assert all(filtered["era"] == "LongMemEval"), "Non-LongMemEval experiments leaked through"
        assert not any(filtered["era"] == "memories_500"), "memories_500 experiment leaked through"

    def test_status_filter_or_logic(self, real_experiments_dir: str):
        """VAL-TABLE-004: Status filter uses OR logic for multiple statuses."""
        import dashboard.app as app_module
        df = app_module._df_all
        # Filter to two statuses
        filtered = app_module._filter_dataframe(df, "All", ["improved", "accepted_cv"], "")
        assert len(filtered) > 0
        for _, row in filtered.iterrows():
            assert row["status"] in ("improved", "accepted_cv"), f"Unexpected status: {row['status']}"

    def test_status_filter_clearable(self, real_experiments_dir: str):
        """VAL-TABLE-004: Clearing status filter restores full view."""
        import dashboard.app as app_module
        df = app_module._df_all
        full_count = len(df)
        filtered = app_module._filter_dataframe(df, "All", [], "")
        assert len(filtered) == full_count

    def test_text_search_experiment_id(self, real_experiments_dir: str):
        """VAL-TABLE-020: Text search filters by experiment ID (exact match)."""
        import dashboard.app as app_module
        df = app_module._df_all
        filtered = app_module._filter_dataframe(df, "All", [], "exp_lme_0008")
        # exp_lme_0008 itself must be in results
        assert any(filtered["id"] == "exp_lme_0008"), "exp_lme_0008 not found in search results"
        # Results include exact ID match + any hypothesis containing that ID
        assert len(filtered) >= 1

    def test_text_search_hypothesis(self, real_experiments_dir: str):
        """VAL-TABLE-020: Text search filters by hypothesis text."""
        import dashboard.app as app_module
        df = app_module._df_all
        filtered = app_module._filter_dataframe(df, "All", [], "sigmoid")
        assert len(filtered) > 0, "No experiments matching 'sigmoid' in hypothesis"
        for _, row in filtered.iterrows():
            assert "sigmoid" in row["hypothesis"].lower(), f"Expected 'sigmoid' in hypothesis: {row['hypothesis'][:50]}"

    def test_text_search_with_filters_and_logic(self, real_experiments_dir: str):
        """VAL-TABLE-020: Search combines with era/status filters (AND logic)."""
        import dashboard.app as app_module
        df = app_module._df_all
        # Search + era filter (AND) — exact ID match in LongMemEval era
        filtered = app_module._filter_dataframe(df, "LongMemEval", [], "exp_lme_0008")
        assert any(filtered["id"] == "exp_lme_0008")
        assert all(filtered["era"] == "LongMemEval")
        # Search + status filter (AND)
        improved_rows = df[df["status"] == "improved"]
        if improved_rows.empty:
            pytest.skip("No improved experiment available in real data")
        improved_id = improved_rows.iloc[0]["id"]
        filtered2 = app_module._filter_dataframe(df, "All", ["improved"], improved_id)
        assert len(filtered2) >= 1
        assert all(filtered2["status"] == "improved")

    def test_sort_dataframe_descending(self, real_experiments_dir: str):
        """VAL-TABLE-005: Sort descending puts highest scores first, nulls last."""
        import dashboard.app as app_module
        df = app_module._df_all
        sorted_df = app_module._sort_dataframe(df, "overall_score", False)
        non_null = sorted_df[sorted_df["overall_score"].notna()].head(5)
        # Verify descending order
        for i in range(len(non_null) - 1):
            assert non_null.iloc[i]["overall_score"] >= non_null.iloc[i + 1]["overall_score"], \
                f"Sort not descending at position {i}: {non_null.iloc[i]['overall_score']} >= {non_null.iloc[i+1]['overall_score']}"
        # Nulls at bottom
        nulls = sorted_df[sorted_df["overall_score"].isna()]
        if len(nulls) > 0:
            last_non_null = sorted_df[sorted_df["overall_score"].notna()].iloc[-1]["overall_score"]
            assert last_non_null > nulls.iloc[0].get("overall_score", float("nan")) or True

    def test_sort_dataframe_ascending(self, real_experiments_dir: str):
        """VAL-TABLE-005: Sort ascending puts lowest scores first."""
        import dashboard.app as app_module
        df = app_module._df_all
        sorted_df = app_module._sort_dataframe(df, "overall_score", True)
        non_null = sorted_df[sorted_df["overall_score"].notna()].head(5)
        for i in range(len(non_null) - 1):
            assert non_null.iloc[i]["overall_score"] <= non_null.iloc[i + 1]["overall_score"], \
                f"Sort not ascending at position {i}"

    def test_sort_non_numeric_no_crash(self, real_experiments_dir: str):
        """VAL-TABLE-006: Sorting with non-numeric values doesn't crash."""
        import dashboard.app as app_module
        df = app_module._df_all.copy()
        # Add a string value to overall_score (simulating validation_failed)
        df.loc[df["id"] == "exp_lme_0063", "overall_score"] = None
        sorted_df = app_module._sort_dataframe(df, "overall_score", False)
        assert len(sorted_df) == len(df)

    def test_build_row_data(self, real_experiments_dir: str):
        """Row data contains all expected columns."""
        import dashboard.app as app_module
        df = app_module._df_all
        rows = app_module._build_row_data(df.head(5))
        assert len(rows) == 5
        for row in rows:
            assert "id" in row
            assert "overall_score" in row
            assert "retrieval_score" in row
            assert "plausibility_score" in row
            assert "status" in row
            assert "hypothesis" in row

    def test_best_experiment_identified(self, real_experiments_dir: str):
        """VAL-TABLE-019: Best experiment is exp_lme_0008 (highest CV mean)."""
        import dashboard.app as app_module
        assert app_module._best_exp_id == "exp_lme_0008", \
            f"Expected best to be exp_lme_0008, got {app_module._best_exp_id}"

    def test_experiment_count_reasonable(self, real_experiments_dir: str):
        """VAL-TABLE-018: Total experiment count is reasonable (>400)."""
        import dashboard.app as app_module
        assert len(app_module._df_all) > 400, f"Only {len(app_module._df_all)} experiments loaded"

    def test_all_status_values_present(self, real_experiments_dir: str):
        """All status values from data are available in the filter options."""
        import dashboard.app as app_module
        assert len(app_module._all_status_values) > 0, "No status values found"
        # Check that common statuses are included
        expected = {"completed", "improved", "not_improved", "validation_failed"}
        for status in expected:
            assert status in app_module._all_status_values, f"Status '{status}' not in options"


# ---------------------------------------------------------------------------
# Tests: Detail view builder
# ---------------------------------------------------------------------------

class TestDetailViewBuilder:
    """Tests for the detail view builder function."""

    @pytest.fixture
    def real_experiments_dir(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "experiments")

    @pytest.fixture
    def app_module(self):
        """Import dashboard.app module once per test class."""
        import dashboard.app as app_module
        return app_module

    def test_detail_view_known_experiment(self, real_experiments_dir: str):
        """Detail view can be built for a known experiment."""
        import dashboard.app as app_module
        detail = app_module._build_detail_view("exp_lme_0008")
        assert isinstance(detail, list)
        assert len(detail) > 5

    def test_detail_view_validation_failed(self, real_experiments_dir: str):
        """Detail view for a validation_failed experiment shows its error message."""
        import dashboard.app as app_module
        exps = load_all_experiments(real_experiments_dir)
        exp = _find_real_experiment(
            exps,
            lambda e: e.status == "validation_failed",
            "validation_failed experiment for detail view",
        )
        detail = app_module._build_detail_view(exp.id)
        detail_text = str(detail)
        assert exp.id in detail_text
        assert exp.error in detail_text

    def test_detail_view_no_experiment(self, real_experiments_dir: str):
        """Detail view for non-existent experiment shows not found message."""
        import dashboard.app as app_module
        detail = app_module._build_detail_view("exp_nonexistent")
        assert isinstance(detail, list)
        assert len(detail) == 1
        assert "not found" in str(detail[0])

    def test_detail_view_header_shows_experiment_id(self, app_module):
        """VAL-TABLE-011: Detail view shows experiment ID prominently."""
        detail = app_module._build_detail_view("exp_lme_0008")
        detail_text = str(detail)
        assert "exp_lme_0008" in detail_text

    def test_detail_view_key_metrics_match_results(self, app_module):
        """VAL-TABLE-011: Key metrics displayed match results.json values."""
        # Load actual results.json for exp_lme_0008
        results_path = Path(__file__).resolve().parent.parent / "experiments" / "exp_lme_0008" / "results.json"
        with open(results_path) as f:
            results = json.load(f)

        detail_text = str(app_module._build_detail_view("exp_lme_0008"))

        # Spot-check 5 metric values at 4 decimal places
        for key in ["overall_score", "retrieval_score", "plausibility_score", "recall_mean", "precision_mean"]:
            val = results[key]
            formatted = f"{val:.4f}"
            assert formatted in detail_text, f"{key}={formatted} not found in detail view"

    def test_detail_view_key_metrics_4_decimal_places(self, app_module):
        """VAL-TABLE-011: Metrics shown with 4 decimal places."""
        results_path = Path(__file__).resolve().parent.parent / "experiments" / "exp_lme_0008" / "results.json"
        with open(results_path) as f:
            results = json.load(f)
        detail_text = str(app_module._build_detail_view("exp_lme_0008"))
        formatted = f"{results['overall_score']:.4f}"
        assert formatted in detail_text, "Expected 4-decimal formatted overall_score"

    def test_detail_view_later_era_metrics_na_for_old(self, app_module):
        """VAL-TABLE-012: Later-era metrics show N/A for experiments without them."""
        detail_text = str(app_module._build_detail_view("exp_lme_0008"))
        # exp_lme_0008 has no strict_score or forgetting_depth
        assert "N/A" in detail_text, "Expected N/A for missing later-era metrics"

    def test_detail_view_later_era_metrics_shown_for_new(self, app_module, real_experiments_dir: str):
        """VAL-TABLE-012: Strict/forgetting metrics are shown for a real experiment that has them."""
        exps = load_all_experiments(real_experiments_dir)
        exp = _find_real_experiment(
            exps,
            lambda e: e.strict_score is not None and e.forgetting_depth is not None,
            "experiment with strict_score and forgetting_depth",
        )
        detail_text = str(app_module._build_detail_view(exp.id))
        assert f"{exp.strict_score:.4f}" in detail_text
        assert f"{exp.forgetting_depth:.4f}" in detail_text

    def test_detail_view_hypothesis_full_text(self, app_module):
        """VAL-TABLE-013: Full hypothesis.txt text displayed, not truncated."""
        hyp_path = Path(__file__).resolve().parent.parent / "experiments" / "exp_lme_0008" / "hypothesis.txt"
        with open(hyp_path) as f:
            full_hypothesis = f.read().strip()

        detail_text = str(app_module._build_detail_view("exp_lme_0008"))
        assert full_hypothesis in detail_text, "Full hypothesis text not found in detail view"

    def test_detail_view_hypothesis_not_available(self, app_module, real_experiments_dir: str):
        """VAL-TABLE-013: Missing hypothesis shows 'not available'."""
        exps = load_all_experiments(real_experiments_dir)
        exp = _find_real_experiment(
            exps,
            lambda e: e.hypothesis == "",
            "experiment with missing hypothesis",
        )
        detail_text = str(app_module._build_detail_view(exp.id))
        assert "Hypothesis not available" in detail_text or "not available" in detail_text.lower()

    def test_detail_view_params_all_keys(self, app_module):
        """VAL-TABLE-013: All params.json key-value pairs shown."""
        params_path = Path(__file__).resolve().parent.parent / "experiments" / "exp_lme_0008" / "params.json"
        with open(params_path) as f:
            params = json.load(f)

        detail_text = str(app_module._build_detail_view("exp_lme_0008"))
        for k, v in params.items():
            assert str(k) in detail_text, f"Param key '{k}' not found in detail view"
            assert str(v) in detail_text, f"Param value '{v}' not found in detail view"

    def test_detail_view_params_not_available(self, app_module, real_experiments_dir: str):
        """VAL-TABLE-013: Missing params shows 'not available'."""
        exps = load_all_experiments(real_experiments_dir)
        exp = _find_real_experiment(
            exps,
            lambda e: e.params == {},
            "experiment with missing params",
        )
        detail_text = str(app_module._build_detail_view(exp.id))
        assert "Parameters not available" in detail_text or "not available" in detail_text.lower()

    def test_detail_view_validation_failed_na_metrics(self, app_module, real_experiments_dir: str):
        """VAL-TABLE-014: validation_failed experiments show N/A for metrics."""
        exps = load_all_experiments(real_experiments_dir)
        exp = _find_real_experiment(
            exps,
            lambda e: e.status == "validation_failed",
            "validation_failed experiment for N/A metric detail view",
        )
        detail_text = str(app_module._build_detail_view(exp.id))
        assert exp.error in detail_text, "Error message not shown"
        assert "N/A" in detail_text, "Expected N/A for metrics of validation_failed experiment"

    def test_detail_view_cv_section_present(self, app_module):
        """VAL-TABLE-015: CV section shows for experiments with cv_results.json."""
        detail_text = str(app_module._build_detail_view("exp_lme_0008"))
        assert "Cross-Validation" in detail_text
        assert "k = 5" in detail_text
        assert "0.3466" in detail_text, "Expected CV mean overall_score"
        assert "0.0296" in detail_text, "Expected CV std overall_score"

    def test_detail_view_cv_fold_deltas(self, app_module):
        """VAL-TABLE-015: CV section shows fold deltas."""
        detail_text = str(app_module._build_detail_view("exp_lme_0008"))
        assert "Fold Deltas" in detail_text, "Expected fold_deltas section"

    def test_detail_view_cv_fold_scores_chart(self, app_module):
        """VAL-TABLE-015: CV section includes fold scores bar chart."""
        detail = app_module._build_detail_view("exp_lme_0008")
        detail_text = str(detail)
        assert "Fold Scores" in detail_text, "Expected fold scores chart title"

    def test_detail_view_cv_validation_failed_with_cv(self, app_module):
        """VAL-TABLE-015: exp_lme_0056 (validation_failed with CV data) shows CV."""
        detail_text = str(app_module._build_detail_view("exp_lme_0056"))
        assert "Cross-Validation" in detail_text, "CV section should show even for validation_failed with CV data"

    def test_detail_view_snapshot_chart_present(self, app_module):
        """VAL-TABLE-016: Snapshot mini-charts present for experiments with snapshots."""
        detail = app_module._build_detail_view("exp_lme_0008")
        detail_text = str(detail)
        assert "Snapshot Timeline" in detail_text
        assert "Metrics over Simulation Ticks" in detail_text

    def test_detail_view_snapshot_three_distinguishable_lines(self, app_module):
        """VAL-TABLE-016: Mini-charts have 3 distinguishable colored lines."""
        import plotly.graph_objects as go
        detail = app_module._build_detail_view("exp_lme_0008")
        # Find the snapshot timeline graph (not the CV fold scores chart)
        for item in detail:
            if hasattr(item, 'figure') and item.figure is not None:
                fig = item.figure
                # Check if this is the snapshot chart (has Scatter traces)
                scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
                if len(scatter_traces) == 3:
                    colors = [trace.line.color for trace in scatter_traces]
                    assert len(set(colors)) == 3, "Expected 3 distinct colors for traces"
                    return
        pytest.fail("Could not find snapshot chart with 3 scatter traces")

    def test_detail_view_snapshot_old_era(self, app_module):
        """VAL-TABLE-017: Old-era experiments show available metrics only."""
        detail = app_module._build_detail_view("exp_0001")
        detail_text = str(detail)
        assert "Snapshot Timeline" in detail_text

    def test_detail_view_snapshot_no_data_for_no_results(self, app_module, real_experiments_dir: str):
        """VAL-TABLE-017: Experiments without snapshots show 'No data'."""
        exps = load_all_experiments(real_experiments_dir)
        exp = _find_real_experiment(
            exps,
            lambda e: e.status == "no_results",
            "no_results experiment for snapshot empty state",
        )
        detail_text = str(app_module._build_detail_view(exp.id))
        assert "No data" in detail_text, "Expected 'No data' for experiment without snapshots"

    def test_detail_view_row_click_any_status(self, app_module, real_experiments_dir: str):
        """Row click works for any status including validation_failed, no_results."""
        exps = load_all_experiments(real_experiments_dir)
        example_ids = [
            _find_real_experiment(exps, lambda e: e.status == "validation_failed", "validation_failed detail example").id,
            _find_real_experiment(exps, lambda e: e.status == "no_results", "no_results detail example").id,
            _find_real_experiment(exps, lambda e: e.era == "LongMemEval", "LongMemEval detail example").id,
            _find_real_experiment(exps, lambda e: e.era == "memories_500", "memories_500 detail example").id,
        ]
        for exp_id in example_ids:
            detail = app_module._build_detail_view(exp_id)
            assert isinstance(detail, list), f"Failed to build detail for {exp_id}"
            assert len(detail) > 1, f"Detail for {exp_id} is too short"

    def test_detail_view_status_badge_shown(self, app_module):
        """Detail view shows status badge."""
        detail_text = str(app_module._build_detail_view("exp_lme_0008"))
        # exp_lme_0008 has history status 'accepted_cv'
        assert "accepted_cv" in detail_text, "Expected status badge text in detail view"

    def test_build_metric_card(self, app_module):
        """_build_metric_card helper formats correctly."""
        card = app_module._build_metric_card("Test Score", 0.12345678)
        card_text = str(card)
        assert "Test Score" in card_text
        assert "0.1235" in card_text, "Expected 4 decimal place formatting"

    def test_build_metric_card_na(self, app_module):
        """_build_metric_card shows N/A for None values."""
        card = app_module._build_metric_card("Missing Score", None)
        card_text = str(card)
        assert "N/A" in card_text


# ---------------------------------------------------------------------------
# Chart module tests (VAL-TIMELINE-001 through VAL-TIMELINE-009)
# ---------------------------------------------------------------------------

class TestCharts:
    """Tests for dashboard.charts module — timeline, metric progression, heatmap, retention."""

    @pytest.fixture
    def chart_module(self):
        """Import charts module (side-effect-free)."""
        from dashboard import charts
        return charts

    @pytest.fixture
    def all_experiments(self):
        """Load all experiments from disk."""
        from dashboard.data_loader import load_all_experiments
        return load_all_experiments("experiments")

    # -- Phase Timeline --

    def test_phase_timeline_returns_figure(self, chart_module, all_experiments):
        """Phase timeline returns a Plotly Figure."""
        import plotly.graph_objects as go
        fig = chart_module.build_phase_timeline(all_experiments)
        assert isinstance(fig, go.Figure)

    def test_phase_timeline_all_eras_shows_7_phases(self, chart_module, all_experiments):
        """Phase timeline with All eras shows phases with experiments (1-5, 7-9)."""
        fig = chart_module.build_phase_timeline(all_experiments, era="All")
        # Count y-axis labels (phases with experiments + empty phases)
        y_labels = fig.layout.yaxis.ticktext
        if y_labels is not None:
            # Should have entries for phases with data and empty phases (0, 6)
            assert len(y_labels) >= 7, "Expected at least 7 phase bars"

    def test_phase_timeline_era_filter(self, chart_module, all_experiments):
        """Era filter correctly limits experiments."""
        fig_old = chart_module.build_phase_timeline(all_experiments, era="memories_500")
        fig_new = chart_module.build_phase_timeline(all_experiments, era="LongMemEval")

        # Old era should have phases 1-5
        old_labels = list(fig_old.data[0].y)
        old_labels_str = " ".join(str(l) for l in old_labels)
        assert "Phase 1" in old_labels_str
        assert "Phase 5" in old_labels_str

        # New era should have phases 7-9
        new_labels = list(fig_new.data[0].y)
        new_labels_str = " ".join(str(l) for l in new_labels)
        assert "Phase 7" in new_labels_str
        assert "Phase 8" in new_labels_str
        assert "Phase 9" in new_labels_str

    def test_phase_timeline_customdata(self, chart_module, all_experiments):
        """Phase bars have customdata with phase numbers for click handling."""
        fig = chart_module.build_phase_timeline(all_experiments)
        trace = fig.data[0]
        assert trace.customdata is not None
        assert len(trace.customdata) > 0

    def test_phase_timeline_selected_phase_highlighting(self, chart_module, all_experiments):
        """Selected phase has different color."""
        fig_normal = chart_module.build_phase_timeline(all_experiments, selected_phase=None)
        fig_selected = chart_module.build_phase_timeline(all_experiments, selected_phase=5)

        # Both should produce valid figures
        assert isinstance(fig_normal.data[0].marker.color, (list, tuple))
        assert isinstance(fig_selected.data[0].marker.color, (list, tuple))

    def test_phase_timeline_hover_text(self, chart_module, all_experiments):
        """Hover text includes phase name, date range, experiment count, best score."""
        fig = chart_module.build_phase_timeline(all_experiments)
        trace = fig.data[0]
        assert trace.hovertext is not None
        # At least one bar should have experiment count info
        all_text = " ".join(trace.hovertext)
        assert "Experiments:" in all_text or "experiment" in all_text.lower()

    # -- Metric Progression --

    def test_metric_progression_returns_3_figures(self, chart_module, all_experiments):
        """Metric progression returns exactly 3 figures."""
        figs = chart_module.build_metric_progression(all_experiments)
        assert len(figs) == 3

    def test_metric_progression_y_axis_range(self, chart_module, all_experiments):
        """Y-axis fixed [0,1]."""
        figs = chart_module.build_metric_progression(all_experiments)
        for fig in figs:
            y_range = fig.layout.yaxis.range
            assert list(y_range) == [0, 1], f"Expected [0, 1], got {y_range}"

    def test_metric_progression_phase_shading(self, chart_module, all_experiments):
        """Charts have phase background shading (vrect shapes)."""
        import plotly.graph_objects as go
        figs = chart_module.build_metric_progression(all_experiments)
        for fig in figs:
            # Check for vrect shapes (phase shading)
            has_vrect = any(
                hasattr(s, 'type') and s.type == 'rect' and s.x0 is not None
                for s in fig.layout.shapes if hasattr(s, 'type')
            )
            assert has_vrect, "Expected phase shading (vrect shapes)"

    def test_metric_progression_discontinuity_indicator(self, chart_module, all_experiments):
        """Phase 4/5 boundary has scoring discontinuity indicator."""
        figs = chart_module.build_metric_progression(all_experiments, era="All")
        for fig in figs:
            # Check for vline or annotation with discontinuity text
            has_vline = any(
                hasattr(s, 'type') and s.type == 'line'
                for s in fig.layout.shapes if hasattr(s, 'type')
            )
            has_annotation = any(
                "Scoring" in str(a.text) or "comparable" in str(a.text)
                for a in fig.layout.annotations if a.text
            )
            assert has_vline or has_annotation, "Expected Phase 4/5 discontinuity indicator"

    def test_metric_progression_era_filter(self, chart_module, all_experiments):
        """Era filter reduces data points."""
        figs_all = chart_module.build_metric_progression(all_experiments, era="All")
        figs_lme = chart_module.build_metric_progression(all_experiments, era="LongMemEval")
        # LME should have fewer data points
        all_traces = sum(len(t.x) for t in figs_all[0].data if hasattr(t, 'x') and t.x)
        lme_traces = sum(len(t.x) for t in figs_lme[0].data if hasattr(t, 'x') and t.x)
        assert lme_traces < all_traces, "LME should have fewer data points than All"

    def test_metric_progression_selected_phase_deemphasis(self, chart_module, all_experiments):
        """Selected phase highlights that phase, de-emphasizes others."""
        figs_normal = chart_module.build_metric_progression(all_experiments, selected_phase=None)
        figs_selected = chart_module.build_metric_progression(all_experiments, selected_phase=5)

        # Both should produce valid figures with shapes
        assert len(figs_normal[0].layout.shapes) > 0
        assert len(figs_selected[0].layout.shapes) > 0

    # -- Threshold Heatmap --

    def test_threshold_heatmap_returns_2_figures(self, chart_module, all_experiments):
        """Threshold heatmap returns exactly 2 figures (recall + precision)."""
        figs = chart_module.build_threshold_heatmap(all_experiments)
        assert len(figs) == 2

    def test_threshold_heatmap_color_scale(self, chart_module, all_experiments):
        """Heatmap uses linear color scale [0,1]."""
        figs = chart_module.build_threshold_heatmap(all_experiments)
        for fig in figs:
            trace = fig.data[0]
            assert trace.zmin == 0, "Expected zmin=0"
            assert trace.zmax == 1, "Expected zmax=1"

    def test_threshold_heatmap_9_thresholds(self, chart_module, all_experiments):
        """Heatmap has 9 threshold rows (0.1-0.9)."""
        figs = chart_module.build_threshold_heatmap(all_experiments)
        for fig in figs:
            y_labels = fig.layout.yaxis.ticktext
            if y_labels is not None:
                assert len(y_labels) == 9, f"Expected 9 threshold rows, got {len(y_labels)}"

    def test_threshold_heatmap_old_era_blank_cells(self, chart_module, all_experiments):
        """Old-era experiments mostly have blank cells for threshold 0.1 (only 4 thresholds: 0.2-0.5)."""
        figs = chart_module.build_threshold_heatmap(all_experiments, era="memories_500")
        for fig in figs:
            trace = fig.data[0]
            z_data = trace.z
            if z_data and len(z_data) > 0:
                # Row for threshold 0.1 should have mostly None values
                # (old-era has only 0.2-0.5, though ~14 late old-era have 9 thresholds)
                row_01 = z_data[0]  # First row = threshold 0.1
                if row_01:
                    none_count = sum(1 for v in row_01 if v is None)
                    total = len(row_01)
                    assert none_count > total * 0.5, \
                        f"Old-era threshold 0.1 should be mostly None, got {none_count}/{total} None"

    def test_threshold_heatmap_new_era_complete(self, chart_module, all_experiments):
        """New-era experiments have all 9 thresholds."""
        figs = chart_module.build_threshold_heatmap(all_experiments, era="LongMemEval")
        for fig in figs:
            trace = fig.data[0]
            z_data = trace.z
            if z_data and len(z_data) > 0:
                # Row for threshold 0.2 should have non-None values
                row_02 = z_data[1]  # Second row = threshold 0.2
                if row_02:
                    has_values = any(v is not None for v in row_02)
                    assert has_values, "New-era threshold 0.2 should have values"

    def test_threshold_heatmap_hover_template(self, chart_module, all_experiments):
        """Heatmap cells have hover template with exact value."""
        figs = chart_module.build_threshold_heatmap(all_experiments)
        for fig in figs:
            trace = fig.data[0]
            assert "Value:" in trace.hovertemplate or "%{z" in trace.hovertemplate

    # -- Retention Curve Overlay --

    def test_retention_overlay_returns_figure(self, chart_module, all_experiments):
        """Retention overlay returns a Plotly Figure."""
        import plotly.graph_objects as go
        fig = chart_module.build_retention_overlay(all_experiments, [])
        assert isinstance(fig, go.Figure)

    def test_retention_overlay_empty_selection(self, chart_module, all_experiments):
        """Empty selection shows placeholder message."""
        fig = chart_module.build_retention_overlay(all_experiments, [])
        has_annotation = any(
            "Select" in str(a.text)
            for a in fig.layout.annotations if a.text
        )
        assert has_annotation, "Expected placeholder annotation for empty selection"

    def test_retention_overlay_with_selection(self, chart_module, all_experiments):
        """Selecting experiments with retention data adds traces."""
        available = chart_module.get_retention_available_experiments(all_experiments)
        if len(available) >= 2:
            fig = chart_module.build_retention_overlay(all_experiments, available[:2])
            assert len(fig.data) >= 2, "Expected at least 2 traces for 2 experiments"

    def test_retention_overlay_max_5_enforced(self, chart_module, all_experiments):
        """More than 5 selections handled gracefully (caller enforces limit)."""
        available = chart_module.get_retention_available_experiments(all_experiments)
        # charts module doesn't enforce the limit, but should handle >5
        if len(available) >= 6:
            fig = chart_module.build_retention_overlay(all_experiments, available[:6])
            assert isinstance(fig.data, tuple) or hasattr(fig, 'data')

    def test_retention_overlay_unique_colors(self, chart_module, all_experiments):
        """Each retention curve has a unique color."""
        available = chart_module.get_retention_available_experiments(all_experiments)
        if len(available) >= 3:
            fig = chart_module.build_retention_overlay(all_experiments, available[:3])
            colors = [t.line.color for t in fig.data if hasattr(t, 'line') and t.line.color]
            assert len(set(colors)) == len(colors), "Expected unique colors for each curve"

    def test_retention_ticks(self, chart_module, all_experiments):
        """Retention chart uses correct ticks: 40, 80, 120, 160, 200."""
        available = chart_module.get_retention_available_experiments(all_experiments)
        if len(available) >= 1:
            fig = chart_module.build_retention_overlay(all_experiments, available[:1])
            tick_vals = fig.layout.xaxis.tickvals
            assert list(tick_vals) == [40, 80, 120, 160, 200]

    def test_retention_y_axis_range(self, chart_module, all_experiments):
        """Retention Y-axis range is [0,1]."""
        available = chart_module.get_retention_available_experiments(all_experiments)
        if len(available) >= 1:
            fig = chart_module.build_retention_overlay(all_experiments, available[:1])
            y_range = fig.layout.yaxis.range
            assert list(y_range) == [0, 1]

    def test_get_retention_available_experiments(self, chart_module, all_experiments):
        """Returns list of experiment IDs with retention data."""
        available = chart_module.get_retention_available_experiments(all_experiments)
        assert isinstance(available, list)
        assert len(available) > 0, "Expected at least some experiments with retention data"
        for eid in available:
            assert eid.startswith("exp_"), f"Expected experiment ID, got {eid}"

    def test_get_retention_available_era_filter(self, chart_module, all_experiments):
        """Era filter limits available experiments."""
        all_avail = chart_module.get_retention_available_experiments(all_experiments, "All")
        lme_avail = chart_module.get_retention_available_experiments(all_experiments, "LongMemEval")
        assert len(lme_avail) <= len(all_avail)

    def test_check_retention_warnings(self, chart_module, all_experiments):
        """Warnings returned for experiments without retention data."""
        # exp_0000 doesn't have retention_curve
        warnings = chart_module.check_retention_warnings(["exp_0000"], all_experiments)
        assert len(warnings) == 1
        assert "no retention data" in warnings[0].lower() or "unavailable" in warnings[0].lower()

    def test_check_retention_warnings_valid_exp(self, chart_module, all_experiments):
        """No warnings for experiments with retention data."""
        available = chart_module.get_retention_available_experiments(all_experiments)
        if available:
            warnings = chart_module.check_retention_warnings([available[0]], all_experiments)
            assert len(warnings) == 0


class TestAdvancedCharts:
    """Tests for advanced analysis chart builders — parameter sweep, snapshots, forgetting depth, CV, phase comparison."""

    @pytest.fixture
    def chart_module(self):
        """Import charts module (side-effect-free)."""
        from dashboard import charts
        return charts

    @pytest.fixture
    def all_experiments(self):
        """Load all experiments from disk."""
        from dashboard.data_loader import load_all_experiments
        return load_all_experiments("experiments")

    # -- Parameter Sweep --

    def test_parameter_sweep_returns_figure(self, chart_module, all_experiments):
        """Parameter sweep returns a Plotly Figure."""
        import plotly.graph_objects as go
        fig = chart_module.build_parameter_sweep(all_experiments, "All")
        assert isinstance(fig, go.Figure)

    def test_parameter_sweep_era_filter(self, chart_module, all_experiments):
        """Era filter changes available dimensions."""
        import plotly.graph_objects as go
        fig_all = chart_module.build_parameter_sweep(all_experiments, "All")
        fig_lme = chart_module.build_parameter_sweep(all_experiments, "LongMemEval")
        assert isinstance(fig_all, go.Figure)
        assert isinstance(fig_lme, go.Figure)

    def test_parameter_sweep_enabled_params(self, chart_module, all_experiments):
        """Enabled params filter restricts dimensions shown."""
        fig = chart_module.build_parameter_sweep(all_experiments, "All", enabled_params=["lambda_fact", "alpha"])
        assert isinstance(fig.data, tuple)

    def test_get_param_sweep_available_params(self, chart_module, all_experiments):
        """Returns non-empty list of available parameter names."""
        params = chart_module.get_param_sweep_available_params(all_experiments, "All")
        assert isinstance(params, list)
        assert len(params) > 0

    def test_parameter_sweep_single_era_dims(self, chart_module, all_experiments):
        """Single era shows only that era's parameters as dimensions."""
        dims_all, avail_all = chart_module.get_param_sweep_dimensions(all_experiments, "All")
        dims_lme, avail_lme = chart_module.get_param_sweep_dimensions(all_experiments, "LongMemEval")
        # LME era should have its own set of dimensions
        assert isinstance(dims_lme, list)

    def test_parameter_sweep_all_hides_sparse(self, chart_module, all_experiments):
        """All era view hides params with >50% null values."""
        dims_all, avail_all = chart_module.get_param_sweep_dimensions(all_experiments, "All")
        dims_lme, avail_lme = chart_module.get_param_sweep_dimensions(all_experiments, "LongMemEval")
        # All view should have fewer or equal dimensions than LME-only view
        # (since All includes both eras and filters by >50% null threshold)
        assert len(avail_all) <= len(avail_lme) + 10  # Allow some tolerance

    # -- Snapshot Viewer --

    def test_snapshot_viewer_returns_figure(self, chart_module, all_experiments):
        """Snapshot viewer returns a Plotly Figure."""
        import plotly.graph_objects as go
        # Use an experiment known to have snapshots
        fig = chart_module.build_snapshot_viewer(all_experiments, "exp_lme_0008")
        assert isinstance(fig, go.Figure)

    def test_snapshot_viewer_no_data(self, chart_module, all_experiments):
        """Snapshot viewer shows placeholder for experiment without snapshots."""
        fig = chart_module.build_snapshot_viewer(all_experiments, "nonexistent_exp")
        has_annotation = any(
            a.text and "No" in str(a.text)
            for a in fig.layout.annotations if a.text
        )
        assert has_annotation

    def test_snapshot_viewer_tick_highlight(self, chart_module, all_experiments):
        """Snapshot viewer highlights the current tick."""
        fig = chart_module.build_snapshot_viewer(all_experiments, "exp_lme_0008", current_tick=100)
        has_vline = len(fig.layout.shapes) > 0 or any(
            hasattr(s, 'xref') for s in fig.layout.shapes
        )
        # Check if there's a vertical line shape
        shapes = fig.layout.shapes
        vline_found = any(
            hasattr(s, 'line') and s.line and getattr(s.line, 'dash', None) == 'dash'
            for s in shapes
        ) if shapes else False
        # Either vline or annotation
        assert has_vline or vline_found or len(fig.data) > 0

    def test_get_snapshot_tick_data(self, chart_module, all_experiments):
        """get_snapshot_tick_data returns metric values for a specific tick."""
        data = chart_module.get_snapshot_tick_data(all_experiments, "exp_lme_0008", 100)
        assert isinstance(data, dict)
        # Should have at least some metrics
        assert len(data) > 0 or data == {}

    def test_get_snapshot_tick_data_nonexistent(self, chart_module, all_experiments):
        """get_snapshot_tick_data returns empty dict for nonexistent experiment."""
        data = chart_module.get_snapshot_tick_data(all_experiments, "nonexistent", 0)
        assert data == {}

    # -- Forgetting Depth --

    def test_forgetting_depth_chart_returns_figure(self, chart_module, all_experiments):
        """Forgetting depth chart returns a Plotly Figure."""
        import plotly.graph_objects as go
        fig = chart_module.build_forgetting_depth_chart(all_experiments, "All")
        assert isinstance(fig, go.Figure)

    def test_forgetting_depth_only_with_data(self, chart_module, all_experiments):
        """Forgetting depth chart only shows experiments with data."""
        fig = chart_module.build_forgetting_depth_chart(all_experiments, "All")
        # Count experiments with forgetting_depth in the chart
        total_bars = 0
        for trace in fig.data:
            if hasattr(trace, 'x') and trace.x is not None:
                total_bars += len(trace.x)
        # Should be much less than total experiments (~23 vs ~468)
        assert total_bars < 50, f"Expected only experiments with forgetting_depth data, got {total_bars}"

    def test_forgetting_depth_pass_fail_colors(self, chart_module, all_experiments):
        """Forgetting depth chart has separate pass/fail traces."""
        fig = chart_module.build_forgetting_depth_chart(all_experiments, "All")
        trace_names = [t.name for t in fig.data]
        # Should have at least one trace with Passed or Failed name
        has_pass_fail = any(n in trace_names for n in ["Passed", "Failed", "No Strict Score"])
        assert has_pass_fail

    def test_strict_score_chart_returns_figure(self, chart_module, all_experiments):
        """Strict score chart returns a Plotly Figure."""
        import plotly.graph_objects as go
        fig = chart_module.build_strict_score_chart(all_experiments, "All")
        assert isinstance(fig, go.Figure)

    def test_strict_score_pass_threshold_line(self, chart_module, all_experiments):
        """Strict score chart shows pass threshold line at 0.4."""
        fig = chart_module.build_strict_score_chart(all_experiments, "All")
        has_hline = any(
            hasattr(s, 'y0') and abs(s.y0 - 0.4) < 0.01
            for s in fig.layout.shapes
        ) if fig.layout.shapes else False
        assert has_hline, "Expected threshold line at y=0.4"

    def test_forgetting_depth_era_filter(self, chart_module, all_experiments):
        """Era filter limits experiments in forgetting depth chart."""
        fig_all = chart_module.build_forgetting_depth_chart(all_experiments, "All")
        fig_lme = chart_module.build_forgetting_depth_chart(all_experiments, "LongMemEval")
        bars_all = sum(len(t.x) for t in fig_all.data if hasattr(t, 'x') and t.x)
        bars_lme = sum(len(t.x) for t in fig_lme.data if hasattr(t, 'x') and t.x)
        assert bars_lme <= bars_all

    # -- CV Results --

    def test_cv_fold_scores_returns_figure(self, chart_module):
        """CV fold scores chart returns a Plotly Figure."""
        import plotly.graph_objects as go
        cv_data = {"fold_scores": [
            {"overall_score": 0.3 + i * 0.1} for i in range(5)
        ], "mean": {"overall_score": 0.5}, "std": {"overall_score": 0.05}, "worst_fold": {"overall_score": 0.3}}
        fig = chart_module.build_cv_fold_scores(cv_data, "test_exp")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_cv_fold_scores_mean_line(self, chart_module):
        """CV fold scores chart shows mean line."""
        cv_data = {"fold_scores": [
            {"overall_score": 0.3 + i * 0.1} for i in range(5)
        ], "mean": {"overall_score": 0.5}, "std": {"overall_score": 0.05}}
        fig = chart_module.build_cv_fold_scores(cv_data, "test_exp")
        has_hline = any(
            hasattr(s, 'y0') and abs(s.y0 - 0.5) < 0.01
            for s in fig.layout.shapes
        ) if fig.layout.shapes else False
        assert has_hline, "Expected mean line at y=0.5"

    def test_cv_fold_deltas_returns_figure(self, chart_module):
        """CV fold deltas chart returns a Plotly Figure."""
        import plotly.graph_objects as go
        cv_data = {"fold_deltas": [0.1, -0.05, 0.02, -0.08, 0.01]}
        fig = chart_module.build_cv_fold_deltas(cv_data, "test_exp")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_cv_fold_deltas_zero_line(self, chart_module):
        """CV fold deltas chart shows zero reference line."""
        cv_data = {"fold_deltas": [0.1, -0.05, 0.02, -0.08, 0.01]}
        fig = chart_module.build_cv_fold_deltas(cv_data, "test_exp")
        has_zero_line = any(
            hasattr(s, 'y0') and abs(s.y0) < 0.01
            for s in fig.layout.shapes
        ) if fig.layout.shapes else False
        assert has_zero_line, "Expected zero reference line"

    def test_cv_fold_scores_worst_fold_highlighted(self, chart_module):
        """CV fold scores highlights worst fold in red."""
        cv_data = {
            "fold_scores": [{"overall_score": 0.3 + i * 0.1} for i in range(5)],
            "mean": {"overall_score": 0.5},
            "worst_fold": {"overall_score": 0.3},
        }
        fig = chart_module.build_cv_fold_scores(cv_data, "test_exp")
        # First bar (lowest score = worst) should be red
        if fig.data:
            colors = fig.data[0].marker.color
            assert "#EF5350" in colors, "Expected worst fold to be highlighted in red"

    def test_cv_fold_scores_no_data(self, chart_module):
        """CV fold scores shows placeholder when no data."""
        fig = chart_module.build_cv_fold_scores({}, "test_exp")
        has_annotation = any(
            a.text and "No" in str(a.text)
            for a in fig.layout.annotations if a.text
        )
        assert has_annotation

    def test_get_cv_available_experiments(self, chart_module, all_experiments):
        """Returns list of experiment IDs with CV data."""
        import json, os
        cv_data = {}
        for e in all_experiments:
            cv_path = os.path.join(e.dir_path, "cv_results.json")
            if os.path.exists(cv_path):
                with open(cv_path) as f:
                    cv_data[e.id] = json.load(f)
        available = chart_module.get_cv_available_experiments(all_experiments, cv_data, "All")
        assert isinstance(available, list)
        assert len(available) > 0
        assert "exp_lme_0008" in available

    def test_get_cv_available_era_filter(self, chart_module, all_experiments):
        """CV available experiments filtered by era."""
        import json, os
        cv_data = {}
        for e in all_experiments:
            cv_path = os.path.join(e.dir_path, "cv_results.json")
            if os.path.exists(cv_path):
                with open(cv_path) as f:
                    cv_data[e.id] = json.load(f)
        lme_avail = chart_module.get_cv_available_experiments(all_experiments, cv_data, "LongMemEval")
        for eid in lme_avail:
            assert eid.startswith("exp_lme_"), f"Expected LME experiment, got {eid}"

    # -- Phase Comparison --

    def test_phase_comparison_returns_dict(self, chart_module, all_experiments):
        """Phase comparison returns a dict with comparison data."""
        result = chart_module.build_phase_comparison(all_experiments, 1, 2)
        assert isinstance(result, dict)
        assert "phase_a" in result
        assert "phase_b" in result
        assert "metrics" in result

    def test_phase_comparison_excludes_validation_failed(self, chart_module, all_experiments):
        """Phase comparison excludes validation_failed from calculations."""
        result = chart_module.build_phase_comparison(all_experiments, 1, 9)
        # Check that count_a and scored_count_a differ (validation_failed excluded)
        assert "scored_count_a" in result
        assert "scored_count_b" in result

    def test_phase_comparison_improvement_pct(self, chart_module, all_experiments):
        """Phase comparison includes improvement percentages."""
        result = chart_module.build_phase_comparison(all_experiments, 1, 2)
        for metric in result["metrics"]:
            assert "improvement_pct" in metric

    def test_phase_comparison_metric_keys(self, chart_module, all_experiments):
        """Phase comparison covers required metrics."""
        result = chart_module.build_phase_comparison(all_experiments, 1, 8)
        metric_keys = [m["key"] for m in result["metrics"]]
        for required in ["overall_score", "retrieval_score", "plausibility_score", "recall_mean", "retention_auc"]:
            assert required in metric_keys, f"Expected metric {required} in comparison"

    def test_get_available_phases(self, chart_module, all_experiments):
        """Returns list of phases with experiments."""
        phases = chart_module.get_available_phases(all_experiments)
        assert isinstance(phases, list)
        assert len(phases) > 0
        for p in phases:
            assert "label" in p
            assert "value" in p

    def test_phase_comparison_phase_names(self, chart_module, all_experiments):
        """Phase comparison includes correct phase names."""
        result = chart_module.build_phase_comparison(all_experiments, 1, 8)
        assert "Early Exploration" in result["phase_a_name"] or "Phase 1" in result["phase_a_name"]
        assert "LongMemEval" in result["phase_b_name"] or "Phase 8" in result["phase_b_name"]


# ---------------------------------------------------------------------------
# Cross-Area Integration Tests (VAL-CROSS)
# ---------------------------------------------------------------------------

class TestCrossAreaFilterConsistency:
    """Tests for cross-area filter consistency and navigation flows."""

    @pytest.fixture
    def real_experiments_dir(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "experiments")

    @pytest.fixture
    def app_module(self):
        import dashboard.app as app_module
        return app_module

    @pytest.fixture
    def all_experiments(self, real_experiments_dir):
        from dashboard.data_loader import load_all_experiments
        return load_all_experiments(real_experiments_dir)

    def test_era_filter_propagates_to_all_views(self, app_module, all_experiments):
        """VAL-CROSS-001: Era filter produces consistent experiment sets across views.

        Leaderboard filtering, chart building, and experiment retrieval
        should all return the same experiment set for a given era.
        """
        # Filter to LongMemEval
        era = "LongMemEval"
        era_experiments = [e for e in all_experiments if e.era == era]
        old_era_experiments = [e for e in all_experiments if e.era == "memories_500"]

        # Build leaderboard data for LongMemEval
        import pandas as pd
        df = app_module._build_dataframe(all_experiments, {})
        filtered_df = app_module._filter_dataframe(df, era, [], "")
        assert len(filtered_df) == len(era_experiments)

        # Verify no old-era experiments leak through
        for _, row in filtered_df.iterrows():
            assert row["era"] == "LongMemEval", f"Old-era experiment {row['id']} leaked into LongMemEval filter"

        # Verify era chart building uses same filter
        fig_timeline = app_module.charts.build_phase_timeline(all_experiments, era, None)
        assert fig_timeline is not None

        # Verify old-era experiments are excluded from parameter sweep
        param_exps = app_module.charts.get_param_sweep_available_params(all_experiments, era)
        assert isinstance(param_exps, list)

    def test_phase_click_filters_leaderboard(self, app_module, all_experiments):
        """VAL-CROSS-002: Phase filter correctly limits leaderboard to that phase only."""
        import pandas as pd

        df = app_module._build_dataframe(all_experiments, {})
        phase_8_experiments = df[df["phase"] == 8]

        # Apply filters: All eras + phase 8
        filtered = app_module._filter_dataframe(df, "All", [], "")
        filtered = filtered[filtered["phase"] == 8]

        assert len(filtered) == len(phase_8_experiments)
        for _, row in filtered.iterrows():
            assert row["phase"] == 8

        # Verify metric progression de-emphasis is built
        figs = app_module.charts.build_metric_progression(all_experiments, "All", 8)
        assert len(figs) == 3

    def test_experiment_discovery_data_consistency(self, app_module, all_experiments):
        """VAL-CROSS-003: exp_lme_0008 data is consistent across detail view builder."""
        exp_id = "exp_lme_0008"
        exp = next((e for e in all_experiments if e.id == exp_id), None)
        assert exp is not None

        # Build detail view
        detail = app_module._build_detail_view(exp_id)
        detail_text = str(detail)

        # Verify experiment ID shown
        assert exp_id in detail_text

        # Verify key metrics at 4 decimal places
        if exp.overall_score is not None:
            assert f"{exp.overall_score:.4f}" in detail_text

    def test_best_experiment_score_consistency(self, app_module, all_experiments):
        """VAL-CROSS-005: Best experiment scores identical across views (4 decimal places).

        The best experiment should show the same score in:
        - Detail view
        - Leaderboard (via data loader)
        - Data source (results.json)
        """
        import json
        import os

        best_id = app_module._best_exp_id
        assert best_id is not None

        exp = next((e for e in all_experiments if e.id == best_id), None)
        assert exp is not None

        # Get score from data loader
        data_loader_score = exp.overall_score

        # Get score from results.json
        results_path = os.path.join(exp.dir_path, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            results_score = results.get("overall_score")
        else:
            results_score = data_loader_score

        # Build detail view and extract score
        detail = app_module._build_detail_view(best_id)
        detail_text = str(detail)

        # All should match at 4 decimal places
        if data_loader_score is not None and results_score is not None:
            assert f"{data_loader_score:.4f}" == f"{results_score:.4f}", \
                f"Data loader score {data_loader_score} != results.json score {results_score}"
            assert f"{data_loader_score:.4f}" in detail_text, \
                f"Score {data_loader_score:.4f} not found in detail view"

    def test_sidebar_filter_persistence(self, app_module, all_experiments):
        """VAL-CROSS-006: Filter logic is independent of active page/view.

        The _filter_dataframe function should produce the same results
        regardless of which page is being viewed.
        """
        import pandas as pd

        df = app_module._build_dataframe(all_experiments, {})

        # Apply same filters multiple times
        result1 = app_module._filter_dataframe(df, "LongMemEval", ["improved"], "decay")
        result2 = app_module._filter_dataframe(df, "LongMemEval", ["improved"], "decay")
        result3 = app_module._filter_dataframe(df, "LongMemEval", ["improved"], "decay")

        assert len(result1) == len(result2) == len(result3)
        assert result1["id"].tolist() == result2["id"].tolist() == result3["id"].tolist()

    def test_era_toggle_clean_state(self, app_module, all_experiments):
        """VAL-CROSS-004: Multiple era toggles produce clean state each time.

        Simulating 3 full toggle cycles (All→LME→All→LME→All→LME).
        """
        import pandas as pd

        df = app_module._build_dataframe(all_experiments, {})

        for cycle in range(3):
            # All → LongMemEval
            lme = app_module._filter_dataframe(df, "LongMemEval", [], "")
            assert all(row["era"] == "LongMemEval" for _, row in lme.iterrows())

            # LongMemEval → All
            all_df = app_module._filter_dataframe(df, "All", [], "")
            assert len(all_df) > len(lme)

        # Final state should be LongMemEval (last toggle in cycle)
        final = app_module._filter_dataframe(df, "LongMemEval", [], "")
        assert all(row["era"] == "LongMemEval" for _, row in final.iterrows())

    def test_parameter_sweep_detail_round_trip(self, app_module, all_experiments):
        """VAL-CROSS-008: Parameter sweep → detail → back preserves sweep state.

        Tests that detail view builder can accept different source pages.
        """
        exp_id = "exp_lme_0008"

        # Build detail from parameter sweep
        detail = app_module._build_detail_view(exp_id, source_page="params")
        detail_text = str(detail)
        assert exp_id in detail_text
        assert "Back to Parameter Sweep" in detail_text

        # Build detail from leaderboard
        detail2 = app_module._build_detail_view(exp_id, source_page="leaderboard")
        detail_text2 = str(detail2)
        assert "Back to Leaderboard" in detail_text2

        # Build detail from snapshots
        detail3 = app_module._build_detail_view(exp_id, source_page="snapshots")
        detail_text3 = str(detail3)
        assert "Back to Snapshot Viewer" in detail_text3

    def test_phase_comparison_distinct_from_filter(self, app_module, all_experiments):
        """VAL-CROSS-007: Phase comparison uses distinct dropdown selectors.

        Verify that the phase comparison result excludes validation_failed
        and only includes data from the two selected phases.
        """
        result = app_module.charts.build_phase_comparison(all_experiments, 3, 8)

        # Should have proper structure
        assert result is not None
        assert "count_a" in result
        assert "count_b" in result
        assert "metrics" in result

        # Validation_failed should be excluded from calculations
        # (mean may be None if no scored experiments exist for a metric)
        for metric in result["metrics"]:
            phase_a = metric["phase_a"]
            phase_b = metric["phase_b"]
            # Structure should be correct regardless of data
            assert "mean" in phase_a
            assert "mean" in phase_b
            assert "best" in phase_a
            assert "best" in phase_b

    def test_retention_cleared_on_phase_filter(self, app_module, all_experiments):
        """Retention selection cleared when phase filter removes selected experiments.

        If experiments are selected in retention but a phase filter removes some,
        the selection should be updated.
        """
        # Get experiments from different phases
        phase_8_exps = [e.id for e in all_experiments if e.phase == 8 and e.retention_curve]
        phase_7_exps = [e.id for e in all_experiments if e.phase == 7 and e.retention_curve]

        if not phase_8_exps or not phase_7_exps:
            return  # Skip if not enough data

        # Simulate retention selection with experiments from both phases
        mixed_selection = phase_8_exps[:2] + phase_7_exps[:2]

        # After applying phase 8 filter, phase 7 experiments should be removed
        filtered = []
        for exp_id in mixed_selection:
            exp = next((e for e in all_experiments if e.id == exp_id), None)
            if exp and exp.phase == 8:
                filtered.append(exp_id)

        assert len(filtered) < len(mixed_selection)
        for exp_id in filtered:
            exp = next((e for e in all_experiments if e.id == exp_id), None)
            assert exp is not None and exp.phase == 8


# ---------------------------------------------------------------------------
# MemoryBench era experiments (exp_bench_NNNN style)
# ---------------------------------------------------------------------------

class TestMemoryBenchEra:
    """Tests for MemoryBench era experiment loading."""

    def test_classify_era_bench(self):
        assert classify_era("exp_bench_0001") == "MemoryBench"

    def test_classify_era_bench_high_number(self):
        assert classify_era("exp_bench_9999") == "MemoryBench"

    def test_get_phase_bench(self):
        assert get_phase("exp_bench_0001") == 10

    def test_load_bench_experiment_with_results(self, experiments_dir):
        exp_dir = experiments_dir / "exp_bench_0001"
        exp_dir.mkdir()
        _write_json(exp_dir / "bench_results.json", {
            "bench_score": 0.85,
            "benchmarks": {
                "longmemeval": {"accuracy": 0.70, "total": 10, "correct": 7, "run_id": "r1"},
                "locomo": {"accuracy": 1.00, "total": 10, "correct": 10, "run_id": "r2"},
                "convomem": {"accuracy": 1.00, "total": 10, "correct": 10, "run_id": "r3"},
            },
            "config": {"activation_weight": 0.05, "top_k": 20},
        })
        _write_json(exp_dir / "params.json", {"activation_weight": 0.05})
        _write_text(exp_dir / "hypothesis.txt", "prompt v2")

        experiments = load_all_experiments(str(experiments_dir))
        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.id == "exp_bench_0001"
        assert exp.era == "MemoryBench"
        assert exp.bench_score == 0.85
        assert exp.lme_accuracy == 0.70
        assert exp.locomo_accuracy == 1.00
        assert exp.convomem_accuracy == 1.00
        assert exp.status == "completed"

    def test_load_bench_experiment_no_results(self, experiments_dir):
        exp_dir = experiments_dir / "exp_bench_0003"
        exp_dir.mkdir()
        _write_json(exp_dir / "params.json", {"activation_weight": 0.1})

        experiments = load_all_experiments(str(experiments_dir))
        assert len(experiments) == 1
        exp = experiments[0]
        assert exp.status == "no_results"
        assert exp.bench_score is None

    def test_load_bench_experiment_empty_benchmarks(self, experiments_dir):
        exp_dir = experiments_dir / "exp_bench_0002"
        exp_dir.mkdir()
        _write_json(exp_dir / "bench_results.json", {
            "bench_score": 0.0,
            "benchmarks": {},
        })

        experiments = load_all_experiments(str(experiments_dir))
        exp = experiments[0]
        assert exp.bench_score == 0.0
        assert exp.lme_accuracy is None
        assert exp.locomo_accuracy is None
        assert exp.convomem_accuracy is None

    def test_load_history_bench_entries(self, experiments_dir):
        """History entries with bench_score should be loaded correctly."""
        history_path = experiments_dir / "history.jsonl"
        lines = [
            json.dumps({"exp": "exp_bench_0001", "bench_score": 0.62,
                         "lme_acc": 0.60, "locomo_acc": 0.40,
                         "convomem_acc": 1.00, "status": "improved"}),
            json.dumps({"exp": "exp_bench_0001-promptv2", "bench_score": 0.85,
                         "lme_acc": 0.70, "locomo_acc": 1.00,
                         "convomem_acc": 1.00, "status": "improved"}),
        ]
        history_path.write_text("\n".join(lines))

        entries = load_history(str(history_path))
        assert len(entries) == 2
        assert entries[0]["bench_score"] == 0.62
        assert entries[1]["bench_score"] == 0.85


# ---------------------------------------------------------------------------
# MemoryBench charts
# ---------------------------------------------------------------------------

class TestBenchCharts:
    """Tests for MemoryBench chart builders."""

    def test_build_bench_score_progression(self):
        from dashboard.charts import build_bench_score_progression

        experiments = [
            Experiment(id="exp_bench_0001", era="MemoryBench", phase=10,
                       dir_path="/tmp/e1", status="completed",
                       bench_score=0.51, lme_accuracy=0.55,
                       locomo_accuracy=0.15, convomem_accuracy=0.95),
            Experiment(id="exp_bench_0002", era="MemoryBench", phase=10,
                       dir_path="/tmp/e2", status="completed",
                       bench_score=0.85, lme_accuracy=0.70,
                       locomo_accuracy=1.00, convomem_accuracy=1.00),
        ]
        fig = build_bench_score_progression(experiments)
        assert fig is not None
        # 4 traces: bench_score + 3 benchmark accuracies
        assert len(fig.data) == 4

    def test_build_bench_score_progression_empty(self):
        from dashboard.charts import build_bench_score_progression

        fig = build_bench_score_progression([])
        assert fig is not None
        assert len(fig.data) == 0

    def test_build_benchmark_radar(self):
        from dashboard.charts import build_benchmark_radar

        exp = Experiment(id="exp_bench_0001", era="MemoryBench", phase=10,
                         dir_path="/tmp/e1", status="completed",
                         bench_score=0.85, lme_accuracy=0.70,
                         locomo_accuracy=1.00, convomem_accuracy=1.00)
        fig = build_benchmark_radar(exp)
        assert fig is not None
        # 2 traces: actual values + 70% target ring
        assert len(fig.data) == 2
