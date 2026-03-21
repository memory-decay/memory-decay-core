# Memory Chain — Round 0008

## Experiment: exp_lme_0059
**Date**: 2026-03-21
**Parent**: round_0007.md

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.3448 |
| retrieval_score | 0.3687 |
| plausibility_score | 0.5679 |
| strict_score | 0.4585 |
| forgetting_depth | 36.8% |
| recall_mean | 0.633 |

## Cache
- Used: `cache_gemini2_batch` (gemini-embedding-2-preview, 3072 dimensions)
- Previous experiments used `cache/` (768 dimensions, ko-sroberta)
- Note: scores are NOT comparable between caches due to different embedding spaces

## Hypothesis
Zero-impact memories must decay below 0.30 at tick 200 to pass strict validation.
The sigmoid floor for impact=0 was keeping activations at ~0.42.
Fix: bypass sigmoid floor for impact=0, use pure exponential with 3x lambda.

## Self-Criticism
- Strict validation PASSED for the first time — zero-impact bypass works as intended
- overall=0.3448 is slightly better than exp_lme_0056 (0.3425) on same cache
- But overall still low; recall=0.634 at tick 200 means ~63% of test memories still retrievable
- The retrieval curve shows sharp drops at ticks 120 and 160 — activation-based ranking is not smooth
- Correlation (0.27) is weak — activation scores don't track recall well

## Key Insight
- Zero-impact bypass (pure exp) works for strict validation
- But this means many memories ARE decaying rapidly, which hurts overall recall
- The real bottleneck is that activation doesn't correlate with recall (only 0.27)

## Decisions Made
- Zero-impact bypass: confirmed effective for strict validation
- Effective_lambda * 3.0 for impact=0 is the right multiplier

## What To Avoid
- Do not increase lambda_fact further — it would hurt recall on medium-impact items
- Do not try to further lower the floor for impact>0 items — already at 0.45 max

## Next Step Direction
Focus on improving activation-recall correlation (currently 0.27).
Possible: adjust reinforcement params so that recalled memories stay active longer,
or try different stability_weight to make activation tracking smoother across ticks.
