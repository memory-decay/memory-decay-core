# Memory Chain — Round 0015

## Experiment: exp_lme_0157
**Date**: 2026-03-21
**Parent**: [round_0014.md](round_0014.md)

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.5604 (fixed split) |
| retrieval_score | 0.3341 |
| plausibility_score | 0.7866 |
| recall_mean | 0.5221 |
| mrr_mean | 0.4176 |
| precision_lift | 0.0000 |
| CV mean | 0.2528 |
| CV std | 0.0951 |
| CV% | 37.6% |

## Hypothesis
Hybrid dual-state storage-fraction sweep:
- Start from exp_lme_0154, which showed the hybrid branch was promising but under-tuned.
- Increase `retrieval_storage_boost_scale` to 0.80 to see whether overall score can recover toward the single-score baseline before threshold discrimination collapses.

## Self-Criticism
- The fixed-split result looked like a breakthrough, but CV decisively rejected it. Mean CV overall (`0.2528`) is far below both the current fixed-split best and the historical robust baseline.
- This indicates the high fixed-split score came from split-specific alignment between the hybrid reinforcement policy and the held-out test memories, not from a genuinely robust policy improvement.
- The hybrid branch is still more promising than pure retrieval-only, but large storage fractions create the same instability pattern seen earlier in retrieval_consolidation experiments: the policy can look excellent on one split while failing badly on others.
- The core lesson is that fixed-split success is no longer informative enough for this branch. The dual-state hybrid family must be judged by CV first, not by peak fixed-split score.

## Decisions Made
- Do not accept `exp_lme_0157` as a breakthrough.
- Treat high storage-fraction hybrid policies as split-sensitive unless proven otherwise by CV.

## What To Avoid
- Calling a fixed-split spike a breakthrough without running CV.
- Pushing `retrieval_storage_boost_scale` upward based only on single-split gains.

## Next Step Direction
The hybrid branch is still the best structural lead, but the next experiments should target robustness, not just fixed-split score.

Best next directions:
1. Test lower storage fractions around the hybrid branch with immediate CV gating.
2. Prefer modest hybrid settings over aggressive storage reinforcement.
3. If CV instability persists across the hybrid family, conclude that the dual-state policy still needs a qualitatively different reinforcement rule.
