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
        """Load exp_0000 and exp_lme_0059 in same batch, verify field differences."""
        exps = load_all_experiments(real_experiments_dir)
        old = next(e for e in exps if e.id == "exp_0000")
        new = next(e for e in exps if e.id == "exp_lme_0059")

        # Eras
        assert old.era == "memories_500"
        assert new.era == "LongMemEval"

        # Old-era: no strict_score, forgetting_depth, etc.
        assert old.strict_score is None
        assert old.forgetting_depth is None
        assert old.retention_curve is None
        assert old.eval_v2_score is None

        # New-era: has these
        assert new.strict_score is not None
        assert new.forgetting_depth is not None
        assert new.retention_curve is not None

        # Params
        assert old.params.get("beta_fact") is not None
        assert new.params.get("floor_max") is not None

    def test_validation_failed_real(self, real_experiments_dir: str):
        """exp_lme_0063: validation_failed, metrics null, error preserved."""
        exps = load_all_experiments(real_experiments_dir)
        exp = next(e for e in exps if e.id == "exp_lme_0063")
        assert exp.status == "validation_failed"
        assert exp.overall_score is None
        assert exp.error is not None
        assert "decay" in exp.error.lower() or "decay" in exp.error

    def test_no_results_real(self, real_experiments_dir: str):
        """exp_0360: empty dir → status=no_results."""
        exps = load_all_experiments(real_experiments_dir)
        exp = next((e for e in exps if e.id == "exp_0360"), None)
        assert exp is not None
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
        """All 10 phases (0-9) defined."""
        assert set(PHASE_RANGES.keys()) == set(range(10))

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
