# Memory Chain — Round 0002

## Experiment: exp_lme_0007
**Date**: 2026-03-20
**Parent**: round_0001.md

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.3177 |
| retrieval_score | 0.3388 |
| plausibility_score | 0.5837 |
| recall_mean | 0.5781 |
| mrr_mean | 0.3551 |
| precision_lift | 0.0035 |
| similarity_recall_rate | 0.5948 |

## Hypothesis
Lower the sigmoid floor slightly from exp_lme_0002 to test whether the current model is still over-protecting memories. This experiment keeps the Jost-plus-sigmoid structure, the midpoint, and the reactivation protocol fixed, and changes only floor_max from 0.45 to 0.40 so the result is attributable to floor strength rather than retrieval-policy drift. The target is a modest gain in MRR or precision_lift without a large recall collapse.

## Self-Criticism
- The hypothesis only half-held: lowering the floor did improve selectivity-related diagnostics, but the gain was too small to offset the retention loss.
- `threshold_discrimination` rose sharply (`0.0131 -> 0.1503`) and `precision_lift` improved (`0.0019 -> 0.0035`), which means the lower floor did create more separation across thresholds.
- But recall and ranking quality both fell (`recall_mean 0.6195 -> 0.5781`, `mrr_mean 0.3684 -> 0.3551`), dragging `retrieval_score` down enough to lose on the main objective.
- Plausibility improved slightly (`0.5745 -> 0.5837`) through better correlation, so the failure mode is not implausible forgetting. It is overpaying for selectivity with too much retrieval loss.
- This suggests `floor_max=0.45` is near the current optimum for the fixed Jost+sigmoid form on LongMemEval. The search surface is now showing a clean trade-off, not an obviously mis-set hyperparameter.

## Decisions Made
- Do not push `floor_max` below `0.45` again without adding a compensating mechanism.
- Treat `exp_lme_0002` as the de facto fixed-split leader until a new run beats it.
- Keep the retrieval policy fixed; the experiment stayed interpretable because only the floor changed.

## What To Avoid
- Repeating lower-floor variants (`floor_max < 0.45`) with the rest of the function unchanged.
- Mixing retrieval-policy changes with floor tuning, which would make attribution muddy again.

## Next Step Direction
- Hold `floor_max=0.45` and explore `sigmoid_k` only, looking for better high-threshold separation without cutting mid-threshold recall.
- If that fails, the next gains likely require a structural change near the floor rather than another global floor reduction.
