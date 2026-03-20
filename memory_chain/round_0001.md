# Memory Chain — Round 0001

## Experiment: exp_lme_0002
**Date**: 2026-03-20
**Parent**: round_0000.md

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.3360 |
| retrieval_score | 0.3589 |
| plausibility_score | 0.5745 |
| recall_mean | 0.6195 |
| mrr_mean | 0.3684 |
| precision_lift | 0.0019 |
| similarity_recall_rate | 0.6209 |

## Hypothesis
Lower sigmoid floor to allow more forgetting per reviewer feedback.

Reviewer noted that exp_lme_0001's floor_max=0.60 was too protective - even highly important memories were locked at 60% activation, which doesn't match human forgetting curves where important memories do decay substantially.

Changes:
- floor_max: 0.60 -> 0.45 (allow important memories to decay to 45% max, not 60%)
- sigmoid_mid: 0.25 -> 0.30 (require higher importance threshold before floor engages)

## Self-Criticism
- The change from floor_max=0.60 to 0.45 resulted in massive improvement: 0.0374 -> 0.336 overall
- This confirms the reviewer's intuition - the original sigmoid floor was far too protective
- The plausibility score (0.5745) is reasonable, suggesting the decay curve is plausible despite faster forgetting
- precision_lift remains very low (0.0019) - the model isn't differentiating distractors well yet

## Decisions Made
- floor_max=0.45 is a good balance - allows meaningful forgetting while still protecting high-importance memories
- sigmoid_mid=0.30 shifts the threshold so only truly important items get floor protection

## What To Avoid
- Raising floor_max back above 0.50 - reviewer feedback confirmed it was too high
- Lowering sigmoid_mid below 0.25 would make floor too easy to trigger

## Next Step Direction
- Explore whether further reducing floor_max (e.g., 0.35) could improve retrieval precision
- Consider adjusting sigmoid_k (steepness) to create sharper vs softer transitions
- The Jost power=4.0 appears effective - don't reduce it
