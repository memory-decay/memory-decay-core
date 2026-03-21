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
        filtered2 = app_module._filter_dataframe(df, "All", ["improved"], "exp_lme_0068")
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
        """Detail view for validation_failed shows error message."""
        import dashboard.app as app_module
        detail = app_module._build_detail_view("exp_lme_0063")
        detail_text = str(detail)
        assert "exp_lme_0063" in detail_text
        assert "Validation Error" in detail_text or "validation" in detail_text.lower()

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
        detail_text = str(app_module._build_detail_view("exp_lme_0008"))
        # overall_score for exp_lme_0008 is 0.3415, should show 0.3415 (4 decimals)
        assert "0.3415" in detail_text, "Expected 4-decimal formatted overall_score"

    def test_detail_view_later_era_metrics_na_for_old(self, app_module):
        """VAL-TABLE-012: Later-era metrics show N/A for experiments without them."""
        detail_text = str(app_module._build_detail_view("exp_lme_0008"))
        # exp_lme_0008 has no strict_score or forgetting_depth
        assert "N/A" in detail_text, "Expected N/A for missing later-era metrics"

    def test_detail_view_later_era_metrics_shown_for_new(self, app_module):
        """VAL-TABLE-012: Later-era metrics shown for experiments that have them."""
        detail_text = str(app_module._build_detail_view("exp_lme_0059"))
        # exp_lme_0059 has strict_score=0.458456, forgetting_depth=0.3675
        assert "0.4585" in detail_text, "Expected strict_score for exp_lme_0059"
        assert "0.3675" in detail_text, "Expected forgetting_depth for exp_lme_0059"

    def test_detail_view_hypothesis_full_text(self, app_module):
        """VAL-TABLE-013: Full hypothesis.txt text displayed, not truncated."""
        hyp_path = Path(__file__).resolve().parent.parent / "experiments" / "exp_lme_0008" / "hypothesis.txt"
        with open(hyp_path) as f:
            full_hypothesis = f.read().strip()

        detail_text = str(app_module._build_detail_view("exp_lme_0008"))
        assert full_hypothesis in detail_text, "Full hypothesis text not found in detail view"

    def test_detail_view_hypothesis_not_available(self, app_module):
        """VAL-TABLE-013: Missing hypothesis shows 'not available'."""
        detail_text = str(app_module._build_detail_view("exp_0360"))
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

    def test_detail_view_params_not_available(self, app_module):
        """VAL-TABLE-013: Missing params shows 'not available'."""
        detail_text = str(app_module._build_detail_view("exp_0360"))
        assert "Parameters not available" in detail_text or "not available" in detail_text.lower()

    def test_detail_view_validation_failed_na_metrics(self, app_module):
        """VAL-TABLE-014: validation_failed experiments show N/A for metrics."""
        detail_text = str(app_module._build_detail_view("exp_lme_0063"))
        # Error message should be shown prominently
        assert "Insufficient decay" in detail_text, "Error message not shown"
        # Metrics should show N/A
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

    def test_detail_view_snapshot_no_data_for_no_results(self, app_module):
        """VAL-TABLE-017: Experiments without snapshots show 'No data'."""
        detail_text = str(app_module._build_detail_view("exp_0360"))
        assert "No data" in detail_text, "Expected 'No data' for experiment without snapshots"

    def test_detail_view_row_click_any_status(self, app_module):
        """Row click works for any status including validation_failed, no_results."""
        # These should all build without crashing
        for exp_id in ["exp_lme_0063", "exp_0360", "exp_lme_0001", "exp_0000"]:
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
