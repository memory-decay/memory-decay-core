# Memory Chain — Round 0004

## Experiment: exp_lme_0009
**Date**: 2026-03-20
**Parent**: round_0003.md

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.3396 |
| retrieval_score | 0.3628 |
| plausibility_score | 0.5737 |
| recall_mean | 0.6260 |
| mrr_mean | 0.3730 |
| precision_lift | 0.0017 |
| similarity_recall_rate | 0.6275 |

## Hypothesis
Nudge sigmoid_mid upward from 0.30 to 0.32 while keeping the validated exp_lme_0008 shape fixed. The sharper sigmoid_k=30 transition improved retrieval, but plausibility fell because correlation weakened. Raising the midpoint slightly should make floor protection activate for a narrower set of memories, which may recover activation-recall alignment without giving back too much of the retrieval gain.

## Self-Criticism
- The tradeoff behaved exactly as expected: plausibility improved (`0.5605 -> 0.5737`) because correlation recovered, but retrieval slipped (`0.3656 -> 0.3628`).
- The recall and MRR losses were small, but the overall gain from plausibility was still not enough to beat exp_lme_0008 on the main objective.
- This means the current optimum appears to sit closer to `sigmoid_mid=0.30` than `0.32` when `sigmoid_k=30` is held fixed.
- The experiment is still useful because it confirms the local tradeoff surface is smooth: midpoint up recovers alignment, midpoint down favors retention.

## Decisions Made
- Keep `exp_lme_0008` as the best current formulation.
- Do not promote `sigmoid_mid=0.32`; it is a near miss, not an improvement.
- Continue treating `sigmoid_k=30` as the important structural gain.

## What To Avoid
- Larger upward midpoint jumps with the same shape, which will likely continue sacrificing retrieval.
- Treating plausibility recovery alone as enough; the objective still prefers the retention/ranking advantage from exp_lme_0008.

## Next Step Direction
- Probe a smaller midpoint move, such as `0.31`, or test a slightly softer steepness around the new best (`sigmoid_k` between 25 and 30) instead of another coarse shift.
- If nearby micro-tuning fails, stop local floor tuning and move to a different mechanism.
