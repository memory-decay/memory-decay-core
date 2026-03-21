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
- Total: 436 experiments (validation contract said 434)
- memories_500 (exp_NNNN): 358
- LongMemEval (exp_lme_NNNN): 78
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
