# Memory Chain — Round 0005

## Experiment: exp_lme_0010
**Date**: 2026-03-20
**Parent**: round_0004.md

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.3395 |
| retrieval_score | 0.3627 |
| plausibility_score | 0.5737 |
| recall_mean | 0.6260 |
| mrr_mean | 0.3727 |
| precision_lift | 0.0017 |
| similarity_recall_rate | 0.6275 |

## Hypothesis
Test a smaller midpoint adjustment between exp_lme_0008 and exp_lme_0009. Raising sigmoid_mid to 0.32 recovered plausibility but gave back too much retrieval, so 0.31 may capture part of the correlation gain while staying closer to the retrieval profile of the validated best run at 0.30.

## Self-Criticism
- The local interpolation idea did not reveal a hidden sweet spot. `0.31` landed almost exactly on the same tradeoff surface as `0.32`, and was actually a hair worse overall.
- Compared with exp_lme_0008, the run again traded away a small amount of retrieval for a plausibility gain. Compared with exp_lme_0009, it improved `precision_lift` slightly but lost a bit more MRR, leaving the total score fractionally lower.
- This is good evidence that the local `sigmoid_mid` neighborhood has already been explored enough to show its shape: moving upward from `0.30` consistently weakens the current objective.

## Decisions Made
- Keep `exp_lme_0008` as the best current experiment.
- Stop pushing `sigmoid_mid` upward in tiny increments around `0.30`; the return is effectively flat-to-negative.

## What To Avoid
- More midpoint-only trials above `0.30` with the same `floor_max=0.45`, `sigmoid_k=30` structure.
- Spending more loop budget on indistinguishable local variations when the metrics already show the direction.

## Next Step Direction
- Either test a slightly softer steepness than `30` while keeping `mid=0.30`, or move to a different mechanism entirely instead of more midpoint micro-tuning.
