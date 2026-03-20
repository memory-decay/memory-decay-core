# Memory Chain — Round 0006

## Experiment: exp_lme_0011
**Date**: 2026-03-20
**Parent**: round_0005.md

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.3394 |
| retrieval_score | 0.3627 |
| plausibility_score | 0.5729 |
| recall_mean | 0.6260 |
| mrr_mean | 0.3724 |
| precision_lift | 0.0019 |
| similarity_recall_rate | 0.6275 |

## Hypothesis
Test whether a slightly softer sigmoid transition improves the exp_lme_0008 tradeoff. The jump from sigmoid_k=20 to 30 improved retrieval enough to win, but plausibility fell. Lowering sigmoid_k to 25 keeps the sharper floor structure in place while relaxing the transition just enough that activation-recall alignment may recover without fully giving back the retention and ranking gains.

## Self-Criticism
- This landed on the same qualitative tradeoff surface as the midpoint-up trials: plausibility and correlation recovered, but retrieval slipped enough that overall stayed below exp_lme_0008.
- Compared with exp_lme_0008, the run gained a small amount of `precision_lift` and correlation, but lost more on recall, MRR, similarity recall, and retention AUC.
- That means both obvious local escape routes from exp_lme_0008 have now been tested:
  `sigmoid_mid` up
  `sigmoid_k` down
- Neither one beats the current best, which is a strong sign that local floor-shape tuning is close to exhausted.

## Decisions Made
- Keep `exp_lme_0008` as best.
- Stop local floor-only tuning around `floor_max=0.45`, `sigmoid_mid=0.30`, `sigmoid_k=30`.

## What To Avoid
- More nearby `sigmoid_k` reductions with the same overall form.
- Burning additional loop budget on tiny floor-shape tweaks that now show the same failure mode repeatedly.

## Next Step Direction
- Move to a different mechanism, such as changing how stability contributes to the floor or how activation influences ranking, instead of more sigmoid-only local search.
