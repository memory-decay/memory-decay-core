---
name: frontend-worker
description: Frontend worker for building Dash dashboard views, charts, and interactive UI components
---

# Frontend Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features requiring Dash UI implementation: layouts, tables, charts, callbacks, interactive components. Use for `frontend-worker` skillName features.

## Required Skills

- `agent-browser`: For manual UI verification of the dashboard in the browser

## Work Procedure

1. **Read mission context**: Read `mission.md` and `AGENTS.md` for boundaries, conventions, and constraints.
2. **Run init**: Execute `.factory/init.sh` to set up the environment.
3. **Run baseline tests**: Execute `python -m pytest tests/test_dashboard.py -v --tb=short`.
4. **Run the Dash dev server**: Start with `PORT=8050 dashboard/.venv/bin/python dashboard/app.py &`. Wait for it to be ready (curl http://localhost:8050 returns 200). Keep the server running for the entire session.
5. **Implement the feature**:
   - Add Dash components (html.Div, dcc.Graph, dash_table.DataTable, etc.) to the app layout.
   - Write callbacks for interactivity (filtering, sorting, navigation).
   - Use Plotly for all charts (go.Figure, px.express, etc.).
   - Follow existing patterns in the codebase (check what other components look like).
   - Ensure all numeric displays use 4 decimal places.
6. **Manual verification with agent-browser**:
   - For each `verificationSteps` item that starts with "agent-browser:", use the `agent-browser` skill to verify.
   - Take screenshots as evidence.
   - Each interactive check = one `interactiveChecks` entry in the handoff.
7. **Stop the dev server**: `lsof -ti :8050 | xargs kill` before finishing. Ensure no zombie processes.
8. **Run tests**: Execute `python -m pytest tests/test_dashboard.py -v --tb=short` to verify no regressions.

## Example Handoff

```json
{
  "salientSummary": "Built leaderboard table with era filtering, status multi-select, text search, column sorting, and detail view overlay. Verified all interactions with agent-browser: era switching (<2s), sorting (asc/desc), status filtering (OR logic), text search, row click → detail view, back navigation preserving state. URL state management works for bookmarkability.",
  "whatWasImplemented": "dashboard/app.py: Added leaderboard DataTable with 7 columns, sidebar controls (era dropdown, status checklist, search input), detail overlay with metrics/params/hypothesis/CV sections, snapshot mini-charts, URL state via dcc.Location, shared filter state via dcc.Store. Callbacks: era-filter, status-filter, text-search, column-sort, row-click-to-detail, back-navigation, URL-sync.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "dashboard/.venv/bin/python -m pytest tests/test_dashboard.py -v", "exitCode": 0, "observation": "All tests passed"},
      {"command": "lsof -ti :8050 | xargs kill", "exitCode": 0, "observation": "Dev server stopped cleanly"}
    ],
    "interactiveChecks": [
      {"action": "Load dashboard at http://localhost:8050", "observed": "Page loaded with leaderboard showing 434 experiments, sidebar with era/status/search controls"},
      {"action": "Select era=LongMemEval from dropdown", "observed": "Table updated to show 76 exp_lme_* experiments only. Count badge updated."},
      {"action": "Select era=All", "observed": "Table restored to 434 experiments. Sort and search preserved."},
      {"action": "Click overall_score column header", "observed": "Table sorted ascending. Clicked again → descending. Arrow indicator visible."},
      {"action": "Filter status=improved+accepted_cv", "observed": "Only improved and accepted_cv experiments visible. Multi-select OR logic works."},
      {"action": "Type 'exp_lme_0008' in search", "observed": "Only that experiment visible. Cleared search restored full view."},
      {"action": "Click exp_lme_0008 row", "observed": "Detail overlay opened with correct metrics, params, hypothesis, CV data, snapshot charts."},
      {"action": "Click 'Back' button in detail", "observed": "Returned to leaderboard with all filters (era=All, status=improved+accepted_cv) preserved."},
      {"action": "Verify URL updated after clicking exp_lme_0008", "observed": "URL contains experiment parameter. Refreshed page restored same detail view."},
      {"action": "Click exp_lme_0063 (validation_failed) row", "observed": "Detail shows error message prominently. All metrics show N/A."}
    ],
    "tests": {
      "added": [
        {"file": "tests/test_dashboard.py", "cases": [
          {"name": "test_era_filter_longmemeval", "verifies": "Era=LongMemEval shows only exp_lme experiments"},
          {"name": "test_status_filter_multiselect", "verifies": "Multi-status OR logic filtering"},
          {"name": "test_text_search_experiment_id", "verifies": "Partial ID match filtering"},
          {"name": "test_default_sort_overall_desc", "verifies": "Default sort by overall_score descending"}
        ]}
      ]
    },
    "discoveredIssues": []
  }
}
```

## When to Return to Orchestrator

- Cannot start Dash dev server (port 8050 in use or dependency issue)
- agent-browser cannot connect to localhost:8050
- Feature requires backend data that doesn't exist yet
- Existing callback patterns in the app are incompatible with the new feature
