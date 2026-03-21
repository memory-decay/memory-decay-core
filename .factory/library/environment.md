# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Python Environment
- **Venv location**: `dashboard/.venv/`
- **Python version**: 3.12.3
- **Key dependencies**: dash 4.0+, plotly 6.6+, pandas 3.0+, pytest 7.0+
- **No external APIs needed** — all data is local JSON files

## Experiment Data
- **Location**: `experiments/` directory in project root
- **Two eras**: `exp_NNNN` (memories_500, ~358 experiments), `exp_lme_NNNN` (LongMemEval, ~76 experiments)
- **Read-only**: Dashboard reads but never modifies experiment data
- **Schema evolution**: 3 versions of results.json (old: 25 keys, new: 33 keys, latest: 38+ keys)

## Dashboard App Import Side Effects
- `dashboard/app.py` loads all experiment data at module level (~430 experiments). Importing this module triggers full data loading.
- Test files work around this by importing `dashboard.app` inside test methods rather than at module level to avoid heavy import-time side effects during test collection.

## Venv Notes
- The venv at `dashboard/.venv/` was created manually with `--system-site-packages` due to `python3-venv` package not being available on this system. `ensurepip` also unavailable.
- If the venv breaks, recreate with: `python3 -m venv --system-site-packages dashboard/.venv` then install deps via the `install` command in services.yaml.

## Known Edge Cases
- `exp_0360`: Empty directory (0 files) — maps to phase 5 (range extended to 9999 to cover any old-era experiments beyond 359)
- `exp_lme_0001`: Has params.json + hypothesis.txt but no results.json
- `exp_lme_0071`: Has only params.json
- `experiments/best`: Symlink to exp_lme_0008
- `experiments/experiments/`: Nested archive directory (must not recurse)
- `archive_memories500/history.jsonl`: Different key names (experiment vs exp, overall_score vs overall)
