# Memory Chain — Round 0014

## Experiment: exp_lme_0154
**Date**: 2026-03-21
**Parent**: [round_0013.md](round_0013.md)

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.2885 |
| retrieval_score | 0.3046 |
| plausibility_score | 0.6482 |
| recall_mean | 0.4742 |
| mrr_mean | 0.3829 |
| precision_lift | 0.0000 |

## Hypothesis
Hybrid dual-state follow-up to exp_lme_0152:
- Keep the exp_lme_0128 decay, floor, and importance-scaled retrieval settings.
- Use retrieval_consolidation_mode=retrieval_with_storage_fraction with retrieval_storage_boost_scale=0.25.
- Rationale: successful recall should strongly reinforce retrieval, weakly reinforce storage, and reinforce stability once. This should keep threshold discrimination above the single-score baseline while recovering more retrieval quality than pure retrieval_only.

## Self-Criticism
- The hybrid policy was directionally correct but still insufficient. It recovered a lot of what `retrieval_only` lost (`overall` 0.2473 -> 0.2885, `recall_mean` 0.4031 -> 0.4742), but it remained well below `exp_lme_0128` (`overall` 0.3262).
- This means the core failure mode was real: pure retrieval-only reinforcement starved storage too hard. Weak storage reinforcement helps, but the current 25% storage fraction is still not enough to compete with the best single-score policy.
- Importantly, `threshold_discrimination` stayed high at `0.2680`, so the hybrid mode preserved the main dual-state advantage while partially repairing retrieval quality.
- The new mechanism is therefore promising but under-tuned, not structurally dead like pure retrieval-only.

## Decisions Made
- `retrieval_with_storage_fraction` is a viable dual-state policy family.
- It is strictly better than the pure `retrieval_only` branch, but it is not yet good enough to beat the best current tuned baseline.

## What To Avoid
- Returning to pure `retrieval_only` sweeps; the hybrid already dominates that branch.
- Assuming that a small amount of storage reinforcement is sufficient without tuning the storage fraction itself.

## Next Step Direction
Stay on the hybrid dual-state branch and tune the storage fraction before changing the decay law again.

Best next moves:
1. Sweep `retrieval_storage_boost_scale` upward from `0.25` toward the boundary where td starts collapsing.
2. Keep `retrieval_consolidation_mode=retrieval_with_storage_fraction` fixed while doing that sweep.
3. Only after that boundary is found, revisit `retrieval_boost` or `activation_weight`.
