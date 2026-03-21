# Memory Chain - Failure Pattern Analysis (LongMemEval)

## Overview

Fresh start with LongMemEval dataset (ICLR 2025). Previous analysis from memories_500.jsonl archived to `archive_memories500/`.

## What Worked (from prior dataset — needs revalidation)

- Jost Law decay with power=4.0 + sigmoid floor was best on old synthetic data
- To be confirmed on LongMemEval

## What Failed (from prior dataset — needs revalidation)

Previous failures may not apply to new dataset. Re-test before ruling out:
- Gompertz floor
- Decoupled impact/stability
- Memory-type-specific Jost curvature
- Dual-sigmoid floor

## New Findings (LongMemEval - Round 6 completed)

### Verified on LongMemEval:
- Jost Law power=4.0 + sigmoid floor CONFIRMED as best structure
- sigmoid_k=30-40 optimal
- floor_max=0.43-0.45 optimal (lower hurt recall)
- sigmoid_mid=0.30 optimal
- activation_weight=0.3 better than default 0.5 (NEW FINDING)
- lambda_fact=0.008-0.009, lambda_episode=0.035-0.040 near-optimal

### What Failed (LongMemEval):
- alpha variations (1.0, 2.0) - no improvement
- hyperbolic decay - retrieval collapsed (0.18)
- stretched exponential - retrieval collapsed (0.18)
- logarithmic floor - worse (0.27)
- quadratic floor - much worse (0.22)
- impact-only floor - worse (0.30)
- stability as floor multiplier (multiplicative or additive) - no improvement
- assoc_boost variations (0.1, 0.5) - worse
- activation_weight extremes (0.15, 0.7) - worse
- jost_power variations (3.5, 4.5) - worse
- Lower/higher lambda from optimal - worse

### Convergence Diagnosis:
The search has converged. 45 experiments run. Best found: exp_lme_0008 (CV=0.3466).
The Jost+sigmoid floor combination with activation_weight=0.3 appears near-optimal.
Further gains likely require changing evaluator/protocol, not just decay params.

## New Findings (Dual-State, 2026-03-21)

### Verified on the dual-state architecture:
- Separating `storage_score` and `retrieval_score` can produce real threshold discrimination without evaluator hacks.
- Retrieval-only consolidation produces very high `threshold_discrimination`, confirming the storage/retrieval split works mechanically.

### What Failed (Dual-State):
- `retrieval_only` consolidation with the current scalar decay family underperformed badly on overall retrieval quality.
- Increasing `retrieval_boost` inside `retrieval_only` did not recover enough `overall_score`, `recall_mean`, or `mrr_mean`.
- `stability_only_direct` was even worse than retrieval-only on overall score.

### What Partially Worked (Dual-State):
- Hybrid retrieval consolidation (`retrieval_with_storage_fraction`) improved substantially over pure `retrieval_only`.
- The hybrid recovered recall and overall score while keeping much higher `threshold_discrimination` than the single-score baseline.

### CRITICAL BUG FIX (2026-03-21): CV was not evaluating dual-state policies

**All prior dual-state CV results were invalid.** `cross_validator.py` did not pass `reactivation_policy="retrieval_consolidation"` to the simulation, so `apply_retrieval_consolidation()` was never called during CV. All dual-state experiments fell back to plain `scheduled_query` during CV, which is why they produced identical fold scores.

**Corrected results (all under same evaluator code):**
- exp_lme_0157 (storage_fraction=0.80): CV=0.4010, CV%=27.7% — **ACCEPTED as new best**
- exp_lme_0156 (storage_fraction=0.60): CV=0.3792, CV%=28.7% — passes
- exp_lme_0155 (storage_fraction=0.40): CV=0.3662, CV%=29.1% — passes
- exp_lme_0008 (baseline, scheduled_query): CV=0.3505, CV%=37.4%
- exp_lme_0161 (BM25 gate): CV=0.3072, CV%=35.1% — genuinely worse

**Retracted conclusions:**
- "CV collapse" across the hybrid family was a measurement artifact, not a real finding.
- "Conservative and mid-range storage fractions still failed CV" — FALSE, they pass under corrected CV.
- "Lexical agreement gates collapse to the same CV regime" — this one is partially true; BM25 gating genuinely underperforms, but for different reasons than previously thought (it restricts reinforcement too aggressively, not because of split sensitivity).

### What Actually Worked (Dual-State, corrected):
- `retrieval_with_storage_fraction` with storage_scale 0.40–0.80 all pass CV acceptance criteria (CV% < 30%).
- Higher storage_scale monotonically improves CV mean: 0.40→0.3662, 0.60→0.3792, 0.80→0.4010.
- The optimal storage_scale boundary has NOT been found yet — 0.80 is the highest tested and still best.

### Updated Convergence Diagnosis:
- The single-score search surface is exhausted (exp_lme_0008 remains best for scheduled_query policy).
- The dual-state hybrid family IS working and has produced a validated new best (exp_lme_0157).
- The search should continue: storage_scale > 0.80 has not been tested, and other hybrid params remain unexplored.
- BM25/lexical gating is a dead end under the current protocol.
- Rank-aware and capped retrieval variants should be re-evaluated with corrected CV before being dismissed.
