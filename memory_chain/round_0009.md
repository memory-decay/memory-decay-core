# Memory Chain — Round 0009

## Experiment: exp_lme_0072
**Date**: 2026-03-21
**Parent**: round_0008.md

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.3449 |
| retrieval_score | 0.3686 |
| plausibility_score | 0.5702 |
| recall_mean | 0.633 |
| mrr_mean | 0.380 |
| precision_lift | 0.005 |
| activation_recall_correlation | 0.272 |

## Hypothesis
Lower reinforcement gains (direct=0.25, assoc=0.15) to improve activation-recall correlation.
Hypothesis was that weaker reactivation would sharpen activation signal.

## Self-Criticism
- Lower reinforcement gains had essentially NO effect on overall_score (0.3449 vs 0.3449)
- Correlation was unchanged (0.272 vs 0.27)
- Plausibility very slightly improved (0.5702 vs 0.5679) but retrieval slightly decreased
- The simulation is NOT sensitive to reinforcement gains — decay dynamics dominate
- The search has converged on cache_gemini2_batch as well

## Decisions Made
- Reinforcement gain tuning NOT effective on this cache
- The plausibility bottleneck (weak correlation ~0.27) is NOT addressable via reinforcement params

## What To Avoid
- Reinforcement gain variations — no sensitivity observed
- Further Jost/sigmoid floor micro-tuning on this cache

## Next Step Direction
**ESCALATION NEEDED**: 73 experiments total (0000-0072). Both caches show convergence.

The search has reached the ceiling of the decay function parameter space:
- Old cache (768d): best CV=0.3466 (exp_lme_0008)
- New cache (3072d gemini-2): best fixed-split=0.3449 (exp_lme_0068)

The bottleneck is activation-recall correlation (~0.27), which decay parameter tuning cannot fix.
This likely requires changing the evaluation protocol, reactivation policy, or dataset.

Per program.md escalation rule: stop and ask for human decision on whether to widen the search surface.