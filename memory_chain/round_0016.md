# Memory Chain — Round 0016

## Experiment: exp_lme_0161
**Date**: 2026-03-21
**Parent**: [round_0015.md](round_0015.md)

## Scores
| Metric | Value |
|--------|-------|
| overall_score | 0.4233 (fixed split) |
| retrieval_score | 0.2580 |
| plausibility_score | 0.5886 |
| recall_mean | 0.4500 |
| mrr_mean | 0.0000 |
| precision_lift | 0.0000 |
| CV mean | 0.2528 |
| CV std | 0.0951 |
| CV% | 37.6% |

## Hypothesis
Margin + BM25 gated hybrid retrieval rule:
- Keep the exp_lme_0128 decay, floor, and importance-scaled retrieval settings.
- Use `retrieval_margin_bm25_fraction`, where storage is reinforced only when the semantic hit is rank-1, has enough score margin over rank-2, and also passes lexical BM25 agreement.
- Goal: reduce ambiguous storage reinforcement and improve robustness without giving up fixed-split retrieval quality.

## Self-Criticism
- The rule produced another strong fixed-split score, but CV collapsed to the same bad regime as the other hybrid variants.
- That means even adding a lexical agreement gate did not solve the underlying generalization problem. The issue is not just that storage reinforcement was too permissive; the fold structure still dominates.
- This is now strong evidence that the current retrieval-policy family can manufacture impressive fixed-split scores while leaving cross-split robustness unchanged.
- The BM25/semantic agreement idea is not useless, but in this protocol it does not fix the structural CV failure mode.

## Decisions Made
- Do not accept `exp_lme_0161`.
- Treat margin/BM25-gated reinforcement as another non-robust fixed-split improvement, not as a validated breakthrough.

## What To Avoid
- More fixed-split-only retrieval-rule tweaks in the same family.
- Assuming lexical gating alone can rescue the hybrid branch.

## Next Step Direction
The current hybrid retrieval-rule family looks exhausted under CV.

Best next directions:
1. Stop this family unless the protocol itself changes.
2. If continuing, optimize directly against CV with a very small number of carefully chosen policies rather than broad fixed-split sweeps.
3. Otherwise escalate: the remaining bottleneck is likely the split protocol or a deeper retrieval architecture change, not another local reinforcement rule tweak.
