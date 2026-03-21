---
name: backend-worker
description: Backend/data-layer worker for building Python modules, data loaders, and test logic
---

# Backend Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features requiring Python module implementation, data loading/parsing, unit tests, or backend logic. Use for `backend-worker` skillName features.

## Required Skills

None

## Work Procedure

1. **Read mission context**: Read `mission.md` and `AGENTS.md` for boundaries, conventions, and constraints.
2. **Run init**: Execute `.factory/init.sh` to set up the environment (venv, dependencies).
3. **Run baseline tests**: Execute `python -m pytest tests/test_dashboard.py -v --tb=short` to see current state.
4. **Write tests FIRST (TDD)**:
   - Write failing tests for the feature in `tests/test_dashboard.py` (or new test file as appropriate).
   - Tests must cover: normal cases, edge cases (missing files, malformed data, schema evolution), and error paths.
   - Run tests to confirm they FAIL (red phase).
5. **Implement to make tests pass**:
   - Write implementation code in the appropriate `dashboard/` module.
   - Run tests after each logical unit of work.
   - Ensure ALL tests pass (green phase).
6. **Verify against expected behavior**: Re-read the feature description's `expectedBehavior` list. Verify each item manually or with a targeted test.
7. **Run lint/typecheck**: `python -m py_compile dashboard/<module>.py` for syntax check.
8. **Do NOT start any long-running processes** (Dash dev server, watch modes). Tests only.

## Example Handoff

```json
{
  "salientSummary": "Implemented data_loader.py with schema-union loading for 3 result eras, phase mapping for 9 git phases, history.jsonl dedup, and archive history normalization. Wrote 18 unit tests covering schema evolution, missing files, malformed JSON, and performance. All tests pass.",
  "whatWasImplemented": "dashboard/data_loader.py: Experiment dataclass, load_all_experiments() with maxdepth-1 discovery, schema-union results parsing (old/new/latest era), params dict-merge, history dedup (last-line-wins), archive key normalization, phase mapping. tests/test_dashboard.py: 18 test cases.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "dashboard/.venv/bin/python -m pytest tests/test_dashboard.py -v", "exitCode": 0, "observation": "18 tests passed, 0 failed"},
      {"command": "dashboard/.venv/bin/python -m py_compile dashboard/data_loader.py", "exitCode": 0, "observation": "Syntax OK"}
    ],
    "interactiveChecks": [],
    "tests": {
      "added": [
        {"file": "tests/test_dashboard.py", "cases": [
          {"name": "test_discover_experiments_maxdepth1", "verifies": "Only top-level exp dirs found, no recursion into archive/experiments subdir"},
          {"name": "test_missing_results_json", "verifies": "Experiment with no results.json gets status=no_results and null metrics"},
          {"name": "test_validation_failed_results", "verifies": "Only status+error results have null metrics and preserved error"},
          {"name": "test_schema_evolution_null_defaults", "verifies": "Old-era exp missing new fields get null, not KeyError"},
          {"name": "test_threshold_metrics_variation", "verifies": "Old-era 4 thresholds and new-era 9 thresholds both load correctly"},
          {"name": "test_params_field_variation", "verifies": "Old-era beta_fact and new-era assoc_boost coexist via dict-merge"},
          {"name": "test_history_string_overall", "verifies": "history.jsonl validation_failed string handled, excluded from mean"},
          {"name": "test_history_dedup", "verifies": "Duplicate entries resolved by last-occurrence-in-file"},
          {"name": "test_archive_history_normalization", "verifies": "archive key mapping: experiment->exp, overall_score->overall"},
          {"name": "test_phase_mapping", "verifies": "Each experiment maps to exactly one phase, no overlaps/gaps"},
          {"name": "test_era_classification", "verifies": "exp_NNNN=memories_500, exp_lme_NNNN=LongMemEval"},
          {"name": "test_cold_load_performance", "verifies": "Full load under 5 seconds"},
          {"name": "test_memory_efficiency", "verifies": "Dataset under 200MB RAM"},
          {"name": "test_malformed_json_handling", "verifies": "Invalid JSON flagged as parse_error, excluded from aggregations"},
          {"name": "test_missing_params_hypothesis", "verifies": "Missing params={}, missing hypothesis=''"},
          {"name": "test_snapshots_structure", "verifies": "11 snapshots with monotonically increasing ticks"},
          {"name": "test_best_symlink_not_double_counted", "verifies": "best symlink not followed during discovery"},
          {"name": "test_numeric_precision", "verifies": "Float precision preserved, int fields parsed as int"}
        ]}
      ]
    },
    "discoveredIssues": []
  }
}
```

## When to Return to Orchestrator

- Feature depends on data structures or APIs from another feature that doesn't exist yet
- Cannot create venv or install dependencies due to environment issues
- Requirements are ambiguous about data handling behavior
