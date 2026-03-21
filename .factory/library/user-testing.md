# User Testing

Testing surface, required tools, and resource cost classification.

## Validation Surface

**Surface**: Browser (agent-browser) at `http://localhost:8050` for UI milestones; Python scripts for data-layer milestone.

**Data-layer testing approach**: The data-layer milestone (`dashboard/data_loader.py`) is a backend Python module with no UI surface. Assertions are validated via Python scripts run with `dashboard/.venv/bin/python`, NOT via browser automation. Flow validators write and execute Python validation scripts that import from `dashboard.data_loader` and test against real experiment data in `experiments/`.

**UI testing setup requirements**:
1. Run `.factory/init.sh` to create venv and install dependencies
2. Start Dash server: `cd /home/roach/.openclaw/workspace/memory-decay && PORT=8050 dashboard/.venv/bin/python dashboard/app.py`
3. Wait for healthcheck: `curl -sf http://localhost:8050` returns 200
4. agent-browser navigates to `http://localhost:8050`

**Current experiment counts** (updated 2026-03-21):
- Total: 451 experiments (dashboard shows; filesystem has 358+94 dirs, one excluded)
- memories_500 (exp_NNNN): 358
- LongMemEval (exp_lme_NNNN): 93 (dashboard shows; filesystem has 94 dirs)
- No-results experiments: exp_0360, exp_lme_0001, exp_lme_0071, exp_lme_0077

**Key test experiments for validation**:
- `exp_lme_0008`: Current best, has CV data, full metrics
- `exp_lme_0059`: Latest era metrics (strict_score, forgetting_depth)
- `exp_lme_0063`: validation_failed (no metrics, only error)
- `exp_lme_0056`: validation_failed BUT has cv_results.json (edge case)
- `exp_0000`: Old-era baseline (4 thresholds, fewer fields)
- `exp_0360`: Empty directory, unmapped phase (known issue: VAL-DATA-011)
- `exp_lme_0001`: Partial data (no results.json)

**Known data-level findings**:
- exp_0360 has no phase mapping (ID 360 exceeds phase 5 max 359) — VAL-DATA-011 failed
- 4 history.jsonl vs results.json discrepancies (exp_lme_0068/0073/0074/0075 re-scored after history written)
- History has 67 raw entries, 64 unique after dedup (3 duplicates: exp_lme_0059, 0060, 0063)
- Archive history: 60 entries with key normalization (experiment→exp, overall_score→overall)

## Validation Concurrency

**Max concurrent validators: 5**

**Rationale**:
- Each Dash instance: ~43MB RAM
- Each agent-browser session: ~300MB RAM
- Dev server: ~200MB (single shared instance)
- System: 31GB total, ~1.4GB used, ~29GB available
- Usable headroom (70%): 29GB × 0.7 = ~20GB
- 5 validators × 300MB = 1.5GB + 200MB server = 1.7GB total — well within budget

**Isolation**: All validators share the same Dash server on port 8050. No per-validator instances needed. Each agent-browser opens a separate browser tab.

## Validation Approach

- **Era filtering**: Check experiment counts per era, verify no cross-era contamination
- **Status filtering**: Test multi-select OR logic, verify badge colors
- **Sorting**: Verify asc/desc toggle, visual indicator, default sort
- **Detail view**: Compare displayed values against raw JSON files
- **Charts**: Spot-check data points against results.json, verify phase shading boundaries
- **Cross-area**: Verify filter propagation across all views after era/phase changes
- **Performance**: Measure operation times with stopwatch or browser DevTools

## Known UI-Level Findings (dashboard-shell milestone)

### dash-ag-grid 33.3.3 Function-based Config Issues
- **cellStyle functions** not executing — status badges render as plain text, no colors in table
- **rowClassRules** not applying CSS classes — best experiment row not highlighted
- Plain object cellStyle (e.g., on hypothesis column) works fine
- Fix: compute styles in Python and pass as plain object configs

### URL State Management
- dcc.Location(refresh=False) strips query params immediately
- restore_from_url callback works (direct URL navigation restores state)
- update_url_state callback fails to maintain URL params
- Same-page overlay detail view creates no browser history entries

### AG Grid Interaction Quirks
- Row clicks sometimes toggle row selection instead of opening detail view
- Pagination buttons require JavaScript click (not agent-browser refs)
- Sort cycling is 3-state (none→asc→desc→none), not 2-state
- Default sort applied at data level, not via AG Grid sort API — no sort indicator on initial load
- Multi-select status filter: clicking option closes dropdown; must click checkbox input via JS

### Plotly Chart Interaction Limitations (timeline-metrics)
- Plotly charts use internal drag overlay (.draglayer with nsewdrag class) that intercepts all mouse events, preventing agent-browser from triggering Plotly clickData callbacks (e.g., clicking phase timeline bars to filter)
- Even hiding the overlay via CSS doesn't work because Plotly's internal pointer tracking requires native pointer events
- Workaround: verify click callback implementation via source code review + unit tests, not browser automation

### Heatmap Null Cell Display (timeline-metrics)
- Null/None heatmap cells are visually blank as expected (correct)
- Hover on null cells shows "Value: 0.0000" due to Plotly %{z:.4f} template formatting NaN
- Minor UX issue — cells correctly communicate missing data visually

### Dash Custom Dropdown (timeline-metrics)
- Uses virtualized rendering with non-standard DOM classes (dash-options-list, dash-dropdown-content)
- Not accessible via standard Playwright accessibility tree
- Requires direct DOM manipulation via JavaScript to find and click options

### Retention Curve Dropdown (timeline-metrics)
- Pre-filters to only show experiments WITH retention data (114 experiments)
- Experiments without retention data (e.g., exp_0001) are excluded from dropdown
- check_retention_warnings() function exists but code path is unreachable through UI

### Performance Baseline (localhost)
- Initial load: ~145ms (< 3s threshold)
- Era switch: ~248ms (< 2s threshold)
- Sort: ~350ms (< 1s threshold)
- Filter: ~350ms (< 1s threshold)
- Detail view open: ~628ms (< 2s threshold)
- Pagination uses 50 rows/page, ~200ms page change
