# Dashboard Pipeline Outputs Design

**Date**: 2026-03-21
**Status**: Draft
**Goal**: Show output-based benchmark and calibration artifacts in the dashboard alongside canonical experiment results, without changing existing writers under `outputs/` and without forcing those artifacts into the `experiments/` contract.

## Problem

The dashboard currently loads only canonical experiment folders under `experiments/`. That works for `exp_NNNN` and `exp_lme_NNNN`, but it ignores the newer pipeline-style artifacts written under `outputs/`, including:

- `outputs/pre_program_pipeline/suite_summary.json`
- `outputs/pre_program_pipeline/*/comparison_summary.json`
- `outputs/*/human_calibration/metrics.json`
- `outputs/human_calibration/metrics.json`

As a result, the dashboard shows only one part of the project's result surface. Users can run the calibration and benchmark pipelines successfully but still cannot inspect those results in the dashboard unless they open JSON files directly.

## Chosen Approach

Add a separate dashboard page for pipeline outputs and load those artifacts through a second, output-specific contract.

This keeps the current experiment views honest. Experiment-centric pages such as the leaderboard, phase timeline, retention curves, parameter sweep, snapshot viewer, and CV analysis remain driven only by `Experiment` rows from `experiments/`. Output artifacts appear on a dedicated page with a summary table and a pipeline-specific detail view.

## Rejected Alternatives

### Treat `outputs/` artifacts as pseudo-experiments

Rejected because pipeline summaries do not meaningfully map to experiment-era concepts such as phase, retention curves, snapshots, CV folds, or parameter sweep dimensions. Coercing them into `Experiment` would make several existing views misleading or broken.

### Add only a small summary widget to the existing leaderboard page

Rejected because it would hide important artifact details and would not scale to multiple datasets or multiple output types. The user asked to see both result surfaces, not just a headline number.

## Architecture Boundaries

### 1. Experiment contract remains unchanged

`dashboard.data_loader` and the `Experiment` dataclass stay responsible only for `experiments/` discovery and parsing.

### 2. Add an output-specific loader

Add a new module, tentatively `dashboard/output_loader.py`, responsible for:

- discovering relevant files under `outputs/`
- parsing pipeline summary artifacts
- parsing standalone human calibration artifacts
- deduplicating suite-derived and nested summary files
- returning output-specific records that the dashboard can render directly

This module should not depend on the experiment loader.

### 3. Add a pipeline-only dashboard surface

`dashboard.app` gains a new top-level page, tentatively `Pipeline Benchmarks`, with:

- a table of discovered output records
- pipeline-only filtering/search state
- a pipeline detail pane or overlay

This page does not reuse experiment-specific charts or callbacks except for shared layout primitives where that is straightforward.

## Output Data Contract

The dashboard should load two record types from `outputs/`.

### Record type: `benchmark_run`

Represents one benchmark comparison run for one dataset, whether it was discovered from a suite summary or from a standalone comparison summary.

Required fields:

- `record_type`: `benchmark_run`
- `record_id`: stable identifier derived from source directory and dataset name
- `suite_name`: suite directory name when loaded from `suite_summary.json`, otherwise `None`
- `dataset_name`: dataset stem such as `memories_50`
- `source_dir`: directory containing the benchmark run artifacts
- `source_file`: path to `suite_summary.json` or `comparison_summary.json`
- `status`: `completed`, `parse_error`, or `incomplete`
- `error`: parse/integrity error text when applicable
- `baseline_overall`
- `baseline_retrieval`
- `baseline_plausibility`
- `calibrated_overall`
- `calibrated_retrieval`
- `calibrated_plausibility`
- `delta_overall`
- `delta_retrieval`
- `delta_plausibility`
- `calibration_valid_nll`
- `calibration_valid_brier`
- `calibration_valid_ece`
- `calibration_test_nll`
- `calibration_test_brier`
- `calibration_test_ece`
- `num_iterations`
- `seed`
- `total_ticks`
- `eval_interval`
- `decay_type`
- `reactivation_policy`
- `embedding_backend`
- `baseline_output_path`
- `calibrated_output_path`
- `best_params_path`
- `raw_inputs`

### Record type: `human_calibration`

Represents one standalone calibration artifact directory.

Required fields:

- `record_type`: `human_calibration`
- `record_id`: stable identifier derived from calibration directory path
- `source_dir`
- `source_file`: path to `metrics.json`
- `status`: `completed`, `parse_error`, or `incomplete`
- `error`
- `nll`
- `brier`
- `ece`
- `num_events`
- `best_params_path`
- `trials_path`

## Discovery Rules

### Benchmark summaries

1. Scan `outputs/` recursively for `suite_summary.json`.
2. For each suite summary, create one `benchmark_run` record per entry in `runs`.
3. Scan `outputs/` recursively for `comparison_summary.json`.
4. Create standalone `benchmark_run` records only when the parent run directory is not already represented by a discovered suite summary.

This avoids double-counting `outputs/pre_program_pipeline/memories_50/comparison_summary.json` when `outputs/pre_program_pipeline/suite_summary.json` already describes that run.

### Human calibration artifacts

1. Scan `outputs/` recursively for `human_calibration/metrics.json`.
2. Also support the top-level calibration layout under `outputs/human_calibration/metrics.json`.
3. Create one `human_calibration` record per unique calibration directory.

Human calibration records are allowed to coexist with benchmark records that reference the same calibration directory. They are distinct record types and serve different viewing needs.

## Sidebar and Page State

The current sidebar is experiment-specific. On the new pipeline page it should switch to a pipeline-oriented control set instead of showing era and experiment status filters.

Pipeline page controls:

- record-type filter: `All`, `benchmark_run`, `human_calibration`
- text search over dataset name, suite name, and source path
- record count summary

Experiment pages keep the existing sidebar behavior unchanged.

## Pipeline Page UI

The new page should present all discovered output records in a compact sortable table.

Recommended columns:

- Type
- Dataset / Calibration
- Suite
- Overall Delta
- Baseline Overall
- Calibrated Overall
- NLL
- Brier
- ECE
- Status
- Source

Behavior:

- Clicking a row opens a pipeline-specific detail view.
- Benchmark rows show score deltas prominently.
- Human calibration rows show calibration metrics prominently.
- Parse errors and incomplete rows stay visible with clear status badges rather than being silently dropped.

## Detail View

The pipeline detail view should be independent of the experiment detail view. It should not reuse `selected-experiment` state.

### Benchmark detail sections

- score cards for baseline, calibrated, and delta metrics
- calibration metrics section using valid/test metrics when available
- run input configuration
- artifact path section for baseline, calibrated, and parameter files

### Human calibration detail sections

- metric cards for `nll`, `brier`, `ece`, and `num_events`
- artifact path section for `best_params.json`, `trials.json`, and `metrics.json`

## Error Handling

Malformed or incomplete output artifacts should remain visible in the pipeline table.

- invalid JSON -> `parse_error`
- missing required top-level keys -> `incomplete`
- missing optional files such as `best_params.json` or `trials.json` -> record remains `completed`, with missing paths rendered as unavailable

An empty `outputs/` discovery should render a clear empty state on the pipeline page instead of an empty broken table.

## Non-Goals

- No live refresh or polling. Restarting the dashboard is sufficient.
- No rewriting or normalizing `outputs/` artifacts into `experiments/`.
- No attempt to project pipeline artifacts into experiment-only charts such as retention, timeline, or parameter sweep.
- No changes to the existing writers in `src/memory_decay/` or `scripts/`.

## Testing Requirements

Add tests covering:

- discovery from `suite_summary.json`
- discovery from standalone `comparison_summary.json`
- discovery from standalone `human_calibration/metrics.json`
- dedup between suite-derived rows and nested comparison summaries
- malformed JSON and incomplete artifact handling
- pipeline page presence in app layout
- page switching between experiment and pipeline pages
- pipeline detail rendering for both record types

## Success Criteria

- Restarting the dashboard after running a pipeline makes the corresponding `outputs/` results visible.
- Experiment pages continue to behave exactly as before for `experiments/` data.
- Nested output summaries are not double-counted.
- Missing or malformed output artifacts surface as visible status rows instead of disappearing silently.
