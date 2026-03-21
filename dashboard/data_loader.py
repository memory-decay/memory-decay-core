"""Data loader for memory decay experiment dashboard.

Loads all experiment data from the experiments/ directory, handling:
- Both eras: exp_NNNN (memories_500) and exp_lme_NNNN (LongMemEval)
- Schema evolution across 3 versions of results.json
- Evolving params.json fields between eras
- history.jsonl with sparse fields and duplicates
- archive_memories500/history.jsonl with different key names

Returns structured Experiment dataclass instances with null-safe field access.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Experiment naming patterns
# ---------------------------------------------------------------------------

_EXP_OLD_RE = re.compile(r"^exp_(\d{4})$")
_EXP_NEW_RE = re.compile(r"^exp_lme_(\d{4})$")

# Directories at maxdepth 1 that match these patterns are experiments
_EXPERIMENT_NAME_RE = re.compile(r"^exp(_lme)?_\d{4}$")

# Non-experiment entries to exclude
_EXCLUDED_ENTRIES = {
    "archive_memories500",
    "experiments",  # nested experiments/experiments/ dir
}


# ---------------------------------------------------------------------------
# Phase mapping: 9 git phases (0-9)
# ---------------------------------------------------------------------------

# Each phase maps to (era_pattern, id_min, id_max) ranges.
# "old" = exp_NNNN, "new" = exp_lme_NNNN
PHASE_RANGES: dict[int, list[tuple[str, int, int]]] = {
    0: [],  # Project Setup (pre-experiment, no experiments)
    1: [("old", 0, 24)],       # Early Exploration
    2: [("old", 25, 50)],      # Reinforcement Redesign
    3: [("old", 51, 82)],      # Auto-Research First 25
    4: [("old", 83, 296)],     # Protocol Fixes & Extended
    5: [("old", 297, 359)],    # Scoring Overhaul
    6: [],                     # Memory Chain (meta-infrastructure, no experiments)
    7: [("new", 0, 0)],        # LongMemEval Integration
    8: [("new", 1, 45)],       # LongMemEval Auto-Research
    9: [("new", 46, 9999)],    # Batch Embedding & Strict
}


def classify_era(exp_id: str) -> str:
    """Classify experiment by naming pattern into era.

    Returns:
        'memories_500' for exp_NNNN, 'LongMemEval' for exp_lme_NNNN.
    """
    if _EXP_NEW_RE.match(exp_id):
        return "LongMemEval"
    if _EXP_OLD_RE.match(exp_id):
        return "memories_500"
    raise ValueError(f"Unknown experiment format: {exp_id}")


def get_phase(exp_id: str) -> Optional[int]:
    """Map experiment ID to its git phase number (0-9).

    Each experiment maps to exactly one phase. Phases 0 and 6 have
    no experiments, so no ID will map to them.

    Returns:
        Phase number (1-9), or None if unmapped.
    """
    m_old = _EXP_OLD_RE.match(exp_id)
    m_new = _EXP_NEW_RE.match(exp_id)

    if m_old:
        num = int(m_old.group(1))
        era = "old"
    elif m_new:
        num = int(m_new.group(1))
        era = "new"
    else:
        return None

    for phase, ranges in PHASE_RANGES.items():
        for r_era, r_min, r_max in ranges:
            if era == r_era and r_min <= num <= r_max:
                return phase
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """Structured experiment data with null-safe field access.

    Fields that may be missing in older experiment schemas default to None.
    """

    # Identity
    id: str
    era: str
    phase: Optional[int]
    dir_path: str

    # Status
    status: str  # completed, no_results, parse_error, validation_failed
    error: Optional[str] = None

    # Primary metrics (from results.json top-level)
    overall_score: Optional[float] = None
    retrieval_score: Optional[float] = None
    plausibility_score: Optional[float] = None
    recall_rate: Optional[float] = None
    recall_mean: Optional[float] = None
    precision_rate: Optional[float] = None
    precision_mean: Optional[float] = None
    mrr_mean: Optional[float] = None
    corr_score: Optional[float] = None
    smoothness_score: Optional[float] = None
    retention_auc: Optional[float] = None
    duration_seconds: Optional[float] = None
    tick: Optional[int] = None

    # Later-era metrics (absent in early experiments)
    eval_v2_score: Optional[float] = None
    strict_score: Optional[float] = None
    forgetting_depth: Optional[float] = None
    forgetting_score: Optional[float] = None
    selectivity_score: Optional[float] = None
    robustness_score: Optional[float] = None

    # Composite structures
    threshold_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    threshold_summary: Optional[dict[str, float]] = None
    retention_curve: Optional[dict[str, float]] = None
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    # Params and hypothesis
    params: dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""


# ---------------------------------------------------------------------------
# History entry type
# ---------------------------------------------------------------------------

# Type alias for history entries (dicts with variable fields)
HistoryEntry = dict[str, Any]


# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def _safe_load_json(path: str) -> tuple[Optional[dict], Optional[str]]:
    """Load JSON file, returning (data, error).

    Returns (None, error_message) on failure instead of raising.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def _safe_load_text(path: str) -> str:
    """Load text file, returning empty string on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, IOError):
        return ""


# ---------------------------------------------------------------------------
# Core loading functions
# ---------------------------------------------------------------------------

def _parse_results(data: dict) -> dict[str, Any]:
    """Extract metrics from results.json dict into flat Experiment kwargs.

    Uses .get() for all fields to handle schema evolution gracefully.
    """
    kwargs: dict[str, Any] = {}

    # Status handling
    status = data.get("status", "unknown")
    kwargs["status"] = status

    if status == "validation_failed":
        # validation_failed: only status + error, everything else null
        kwargs["error"] = data.get("error")
        return kwargs

    # Standard metrics
    kwargs["overall_score"] = data.get("overall_score")
    kwargs["retrieval_score"] = data.get("retrieval_score")
    kwargs["plausibility_score"] = data.get("plausibility_score")
    kwargs["recall_rate"] = data.get("recall_rate")
    kwargs["recall_mean"] = data.get("recall_mean")
    kwargs["precision_rate"] = data.get("precision_rate")
    kwargs["precision_mean"] = data.get("precision_mean")
    kwargs["mrr_mean"] = data.get("mrr_mean")
    kwargs["corr_score"] = data.get("corr_score")
    kwargs["smoothness_score"] = data.get("smoothness_score")
    kwargs["retention_auc"] = data.get("retention_auc")
    kwargs["duration_seconds"] = data.get("duration_seconds")

    # Tick: ensure int if present
    tick = data.get("tick")
    kwargs["tick"] = int(tick) if tick is not None else None

    # Later-era metrics (null for older experiments)
    kwargs["eval_v2_score"] = data.get("eval_v2_score")
    kwargs["strict_score"] = data.get("strict_score")
    kwargs["forgetting_depth"] = data.get("forgetting_depth")
    kwargs["forgetting_score"] = data.get("forgetting_score")
    kwargs["selectivity_score"] = data.get("selectivity_score")
    kwargs["robustness_score"] = data.get("robustness_score")

    # Composite structures
    kwargs["threshold_metrics"] = data.get("threshold_metrics", {})
    kwargs["threshold_summary"] = data.get("threshold_summary")
    kwargs["retention_curve"] = data.get("retention_curve")

    # Snapshots
    snapshots = data.get("snapshots", [])
    if snapshots is None:
        snapshots = []
    kwargs["snapshots"] = snapshots

    return kwargs


def load_experiment(exp_dir: str) -> Experiment:
    """Load a single experiment from its directory.

    Handles missing files, malformed JSON, and validation_failed results.
    """
    exp_name = os.path.basename(exp_dir)
    era = classify_era(exp_name)
    phase = get_phase(exp_name)

    # Try loading results.json
    results_path = os.path.join(exp_dir, "results.json")
    results_data, parse_error = _safe_load_json(results_path)

    if parse_error is not None:
        # Malformed JSON
        return Experiment(
            id=exp_name,
            era=era,
            phase=phase,
            dir_path=exp_dir,
            status="parse_error",
            error=parse_error,
        )

    if results_data is None:
        # No results.json
        params_data, _ = _safe_load_json(os.path.join(exp_dir, "params.json"))
        hypothesis = _safe_load_text(os.path.join(exp_dir, "hypothesis.txt"))
        return Experiment(
            id=exp_name,
            era=era,
            phase=phase,
            dir_path=exp_dir,
            status="no_results",
            params=params_data or {},
            hypothesis=hypothesis,
        )

    # Parse results
    kwargs = _parse_results(results_data)

    # Load params
    params_path = os.path.join(exp_dir, "params.json")
    params_data, params_error = _safe_load_json(params_path)

    if params_error is not None:
        kwargs["status"] = "parse_error"
        kwargs["error"] = params_error
    else:
        kwargs["params"] = params_data or {}

    # Load hypothesis
    kwargs["hypothesis"] = _safe_load_text(os.path.join(exp_dir, "hypothesis.txt"))

    return Experiment(
        id=exp_name,
        era=era,
        phase=phase,
        dir_path=exp_dir,
        **kwargs,
    )


def load_all_experiments(experiments_dir: str) -> list[Experiment]:
    """Load all experiments from the experiments/ directory.

    Discovers exp_NNNN and exp_lme_NNNN directories at maxdepth 1.
    Excludes non-experiment entries (archive_memories500, best symlink,
    history.jsonl, strict_eval.py, experiments/experiments/).

    Args:
        experiments_dir: Path to the experiments/ directory.

    Returns:
        List of Experiment instances, one per experiment directory.
    """
    experiments: list[Experiment] = []

    for entry in sorted(os.listdir(experiments_dir)):
        entry_path = os.path.join(experiments_dir, entry)

        # Skip non-directories and symlinks (best → exp_lme_0008)
        if not os.path.isdir(entry_path):
            continue

        # Skip excluded directories
        if entry in _EXCLUDED_ENTRIES:
            continue

        # Must match experiment naming pattern
        if not _EXPERIMENT_NAME_RE.match(entry):
            continue

        experiments.append(load_experiment(entry_path))

    return experiments


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------

def load_history(history_path: str) -> list[HistoryEntry]:
    """Load history.jsonl with dedup (last-occurrence-wins).

    Handles:
    - overall field as float or 'validation_failed' string
    - strict_score as float or 'failed'/'validation_failed' string
    - Sparse fields (missing → absent from dict, not None)
    - Duplicate entries (same exp key) deduped by last occurrence in file

    Args:
        history_path: Path to history.jsonl file.

    Returns:
        List of history entry dicts, one per unique experiment.
    """
    if not os.path.exists(history_path):
        return []

    entries: dict[str, HistoryEntry] = {}

    with open(history_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            exp_key = entry.get("exp")
            if exp_key is None:
                continue

            # Last occurrence wins (deterministic dedup)
            entries[exp_key] = entry

    return list(entries.values())


def load_archive_history(archive_path: str) -> list[HistoryEntry]:
    """Load archive_memories500/history.jsonl with key normalization.

    Key mappings applied:
    - experiment → exp
    - overall_score → overall
    - retrieval_score → retrieval
    - plausibility_score → plausibility

    Dedup: last-occurrence-wins, same as load_history.

    Args:
        archive_path: Path to archive history.jsonl file.

    Returns:
        List of history entry dicts with normalized keys.
    """
    if not os.path.exists(archive_path):
        return []

    KEY_MAP = {
        "experiment": "exp",
        "overall_score": "overall",
        "retrieval_score": "retrieval",
        "plausibility_score": "plausibility",
    }

    entries: dict[str, HistoryEntry] = {}

    with open(archive_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Normalize keys
            entry: HistoryEntry = {}
            for key, value in raw.items():
                normalized = KEY_MAP.get(key, key)
                entry[normalized] = value

            exp_key = entry.get("exp")
            if exp_key is None:
                continue

            # Last occurrence wins
            entries[exp_key] = entry

    return list(entries.values())
