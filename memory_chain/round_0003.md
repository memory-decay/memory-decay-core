# Memory Chain — Round 0003

## Experiment: exp_lme_0008
**Date**: 2026-03-20
**Parent**: round_0002.md

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.3415 |
| retrieval_score | 0.3656 |
| plausibility_score | 0.5605 |
| recall_mean | 0.6318 |
| mrr_mean | 0.3744 |
| precision_lift | 0.0018 |
| similarity_recall_rate | 0.6340 |
| CV overall | 0.3466 +/- 0.0296 |
| CV% | 8.5% |

## Hypothesis
Sharpen the sigmoid floor transition while keeping the successful exp_lme_0002 floor height fixed. Because LongMemEval answer-bearing memories carry higher impact than distractors, increasing sigmoid_k from 20 to 30 should keep target memories near the same floor while pushing low-importance distractors closer to zero floor. The intended outcome is better ranking selectivity and threshold separation without the broad recall loss seen when floor_max was reduced globally.

## Self-Criticism
- The win came from retrieval, not from stricter pruning. `recall_mean`, `mrr_mean`, `similarity_recall_rate`, and `retention_auc` all improved over exp_lme_0002, while `precision_lift` dipped slightly.
- So the sharper sigmoid did not create a big selectivity breakthrough. Instead, it seems to have protected high-importance targets more cleanly than the softer transition, which is enough under the current scoring rule.
- Plausibility softened (`0.5745 -> 0.5605`) because correlation dropped, but the retrieval gain outweighed it.
- The 5-fold CV mean (`0.3466`) exceeded the fixed-split baseline level and the variation stayed low (`8.5%`), so this is not behaving like a fragile one-split accident.
- Eval v2 varied more across folds than overall, so the current improvement is real but narrow: stronger retention/ranking under the existing objective, not a broad dominance across every diagnostic.

## Decisions Made
- Accept `exp_lme_0008` as the current best validated candidate in this LongMemEval loop.
- Keep `floor_max=0.45`; the productive change was sharpening the transition, not lowering the floor.
- Use `sigmoid_k=30` as the new reference point for follow-up work.

## What To Avoid
- Returning to `floor_max < 0.45` without a new compensating mechanism.
- Assuming selectivity was solved. The gain here did not come from a meaningful `precision_lift` jump.

## Next Step Direction
- Hold `sigmoid_k=30` and probe small `sigmoid_mid` adjustments around the current setting to recover some plausibility without giving back the retrieval gain.
- If those micro-tunes flatten out, the next improvements likely require a structural change in how activation predicts recall rather than more floor-only tuning.
