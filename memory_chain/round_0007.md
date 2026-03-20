# Memory Chain — Round 0007

## Experiment: exp_lme_0045
**Date**: 2026-03-20
**Parent**: round_0006.md

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.2208 |
| retrieval_score | 0.2332 |
| plausibility_score | 0.6454 |
| recall_mean | 0.458 |
| mrr_mean | 0.258 |
| precision_lift | -0.001 |

## Hypothesis
Explored quadratic floor: floor = floor_max * importance^2

## Self-Criticism
- Quadratic floor performed very poorly (0.22 overall)
- Emphasizing high-importance too much destroyed discrimination
- The sigmoid's gradual transition is important - quadratic is too aggressive

## Decisions Made
- Quadratic floor NOT viable
- Conclude that the sigmoid transition is optimal for this problem

## What To Avoid
- Quadratic or other aggressive (power > 1) floor functions
- Over-emphasizing importance at expense of transition smoothness

## Next Step Direction
**ESCALATION NEEDED**: 45 experiments run. Search has converged. Best is exp_lme_0008 (CV=0.3466).

The Jost+sigmoid floor mechanism appears near-optimal. Further improvement likely requires:
1. Changing the reactivation policy (currently only train memories get reactivated)
2. Changing evaluation metrics or weights
3. Different dataset or embedding model

Per program.md escalation rule: "If the best next move appears to require changing any file outside the allowed search surface: stop and ask a human decision."
