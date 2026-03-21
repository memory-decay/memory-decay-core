# Memory Chain — Round 0017

## Experiment: exp_lme_0157 (re-evaluation after CV bugfix)
**Date**: 2026-03-21
**Parent**: [round_0016.md](round_0016.md)

## Scores (corrected CV)
| Metric | Value |
|--------|-------|
| overall_score | 0.5604 (fixed split) |
| retrieval_score | 0.3341 |
| plausibility_score | 0.7866 |
| recall_mean | 0.5221 |
| mrr_mean | 0.4176 |
| precision_lift | 0.0000 |
| CV mean | 0.4010 |
| CV std | 0.1109 |
| CV% | 27.7% |

## Critical Bug Found

`cross_validator.py` was not passing `reactivation_policy="retrieval_consolidation"` to `run_experiment_with_split`. This caused ALL dual-state experiments to fall back to plain `scheduled_query` during CV, making `apply_retrieval_consolidation()` never execute. Since experiments 0155–0161 shared the same base decay function (from exp_lme_0128) and only differed in consolidation params, they all produced **identical** CV fold scores: `[0.4216, 0.2051, 0.1978, 0.2298, 0.2095]`.

The fix: detect `retrieval_consolidation_mode` in params.json and pass the correct policy through.

## Self-Criticism
- The "CV collapse" conclusion in rounds 0015 and 0016 was completely wrong. It was a measurement bug, not a fundamental property of the search space.
- The telltale sign — four different experiments producing **exactly identical** CV fold scores to 4 decimal places — should have been caught immediately. No legitimate experiment variation can produce this.
- All strategic conclusions derived from the bugged CV (stopping the dual-state family, declaring the search exhausted) were premature.

## Decisions Made
- Accept exp_lme_0157 as new best (CV mean 0.4010 > baseline 0.3505, CV% 27.7% < 30%).
- Update `experiments/best` symlink to exp_lme_0157.
- Retract all CV-based rejection conclusions for dual-state experiments 0155–0161.
- Retract the "search exhausted" conclusion from rounds 0015–0016.

## What To Avoid
- Accepting identical CV results across meaningfully different experiments without investigation.
- Drawing convergence conclusions from bugged measurements.

## Next Step Direction
The dual-state hybrid family is alive and productive. Immediate next moves:
1. Test storage_scale > 0.80 (0.90, 0.95, 1.0) — the monotonic improvement trend hasn't plateaued.
2. Re-evaluate rank-aware and capped retrieval variants with corrected CV (they may also have been incorrectly rejected).
3. Continue tuning the hybrid branch: `activation_weight`, `retrieval_boost`, and decay params are all open axes.
