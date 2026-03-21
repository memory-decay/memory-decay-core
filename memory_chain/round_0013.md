# Memory Chain — Round 0013

## Experiment: exp_lme_0153
**Date**: 2026-03-21
**Parent**: [round_0010.md](round_0010.md)

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.2473 |
| retrieval_score | 0.2620 |
| plausibility_score | 0.6262 |
| recall_mean | 0.4031 |
| mrr_mean | 0.3359 |
| precision_lift | 0.0000 |

## Hypothesis
Dual-state follow-up to exp_lme_0152:
- Keep the retrieval_only dual-state architecture from 0152.
- Increase retrieval_boost from 0.20 to 0.30.
- Rationale: in retrieval_only mode, successful recall no longer inflates storage, so a larger retrieval-state boost may recover ranking quality and overall score while preserving threshold discrimination.

## Self-Criticism
- The hypothesis failed. Raising `retrieval_boost` inside `retrieval_only` barely helped retrieval quality relative to `0152`, while overall stayed far below `0128`.
- This means the failure mode is not "retrieval_only is underpowered by one scalar boost". The deeper problem is that removing storage reinforcement causes recall eligibility itself to collapse faster than retrieval-only boosting can compensate.
- `threshold_discrimination` remained extremely high (`0.4706`), which confirms the dual-state split is working mechanically. The issue is that the retrieval-only branch produces too little retained storage mass to support strong overall retrieval metrics.
- Compared with `0128`, the experiment traded away too much `recall_mean`, `mrr_mean`, and `precision_strict`. The extra td is not useful at that cost.

## Decisions Made
- `retrieval_only` dual-state with higher `retrieval_boost` is not enough to beat the best single-branch dual-state baseline.
- The useful insight is architectural, not parametric: dual-state separation can create threshold spread, but retrieval-only consolidation is too weak a policy family on its own.

## What To Avoid
- More micro-tuning of `retrieval_boost` inside the current `retrieval_only` mode without adding a stronger storage-preserving mechanism.
- Assuming activation-side retrieval boosts alone can replace storage reinforcement.

## Next Step Direction
Stay inside the dual-state architecture, but stop tuning plain `retrieval_only`.

Best next directions:
1. Introduce an experiment-local policy mode that weakly reinforces storage while strongly reinforces retrieval, instead of the current binary split.
2. If policy modes must remain fixed, compare `activation_and_stability` dual-state runs against retrieval-focused ranking params rather than pushing further on retrieval-only.
3. Do not spend more cycles on pure `retrieval_only` + `retrieval_boost` sweeps.
