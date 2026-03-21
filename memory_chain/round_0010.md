# Memory Chain — Round 0010

## Experiments: exp_lme_0073, exp_lme_0074, exp_lme_0075
**Date**: 2026-03-21
**Parent**: round_0009.md

## Results Summary
| Exp | Overall | Retrieval | Plausibility | Correlation | CV (5-fold) | Status |
|-----|---------|-----------|--------------|-------------|-------------|--------|
| 0073 Hebbian-Decay | 0.3454 | 0.3700 | 0.5560 | 0.243 | 0.3464 ± 0.029 (8.3%) | not_improved |
| 0074 Hyperbolic Floor | 0.2654 | 0.2864 | 0.5116 | 0.3323 | — | validation_failed |
| 0075 Adaptive Threshold | 0.3449 | 0.3686 | 0.5702 | 0.2715 | — | not_improved |
| 0076 retrieval_consolidation | 0.6156 | 0.395 | 0.836 | 0.905 | 0.3409 ± 0.1294 (38%) | not_improved (CV reject) |

## Self-Criticism
- **exp_lme_0073**: Fixed-split +0.0005 over baseline (0.3454 vs 0.3449). CV=0.3464 vs baseline 0.3466 — essentially identical within noise. The distance-from-floor modulation did NOT improve correlation (0.243 vs 0.27 baseline — actually worse). The +0.0005 was a statistical fluke.
- **exp_lme_0074**: Hyperbolic floor catastrophically failed (0.2654). The concave shape gave too-high floors to low-importance items, destroying discrimination. But notably correlation was HIGHER (0.3323 vs 0.27) — suggesting a different mechanism was at play. However overall was far too low.
- **exp_lme_0075**: Two-phase decay tied baseline exactly (0.3449). No improvement.
- **Key insight**: Higher correlation (exp_lme_0074: 0.3323) does NOT guarantee higher overall. The hyperbolic floor decayed everything too aggressively regardless of importance, making recall collapse.

## Decisions Made
- None of the three alternative mechanisms broke through the 0.3449 ceiling
- The search has definitively converged on cache_gemini2_batch
- exp_lme_0073 CV=0.3464 ≈ exp_lme_0008 CV=0.3466 — same ceiling, different cache

## Additional Finding: retrieval_consolidation (exp_lme_0076)
- **Fixed-split: 0.6156** — massive improvement due to additive formula + retrieval-induced facilitation
- **Correlation: 0.90** (vs ~0.27 baseline) — testing effect works exactly as cognitive science predicts
- **BUT CV: 0.3409 ± 0.1294 (38%)** — above 30% threshold, reject
- High fold variance: some folds hit 0.515, others 0.239 — policy is extremely split-sensitive
- The additive formula amplified the improvement but also the variance

## What To Avoid
- Hyperbolic floor variants — far too aggressive on low-importance items
- Distance-from-floor lambda modulation — no benefit, complexity not justified
- Two-phase adaptive threshold — no benefit over existing Jost+sigmoid

## Additional Finding: exp_lme_0077 (retrieval_consolidation + multiplicative)
- CV: 0.3459 ± 0.0310 (9.0%) — essentially tied with best known (0.3466)
- Fixed-split: 0.3905 (+13% over 0.3449) but CV not improved
- retrieval_consolidation boosts correlation to 0.90+ but overall ceiling remains
- Policy-level change not sufficient to break through the decay-function ceiling

## Next Step Direction
**Key insight from exp_lme_0076**: retrieval_consolidation is EXTREMELY split-sensitive (CV%=38). On some splits it hits 0.515, others 0.239. The testing effect works but isn't reliable across train/test boundaries.

**Next experiment (0077)**: Try `retrieval_consolidation` with MULTIPLICATIVE formula (not additive) to get a fair comparison with the baseline. Run with old formula to isolate the policy effect from the formula effect.

If multiplicative formula + retrieval_consolidation shows improvement, explore whether decay params can reduce the fold variance.

## Additional Findings: exp_lme_0078-0084
| Exp | Policy | Key Params | Overall | Plausibility | Status |
|-----|--------|------------|---------|--------------|--------|
| 0078 retrieval_consolidation | retrieval_consolidation | test_react=40/10, retrieval_boost=0.10 | 0.3882 | 0.91 | not_improved |
| 0079 retrieval_consolidation | retrieval_consolidation | retrieval_boost=0.20 | 0.3906 | 0.92 | not_improved |
| 0080 retrieval_consolidation | retrieval_consolidation | stability_decay=0.002 | 0.3868 | 0.92 | not_improved |
| 0081 retrieval_consolidation | retrieval_consolidation | stability_cap=3.0 | 0.3906 | 0.93 | not_improved |
| 0082 retrieval_consolidation | retrieval_consolidation | gains=0.50/0.40 | 0.3888 | 0.90 | not_improved |
| 0083 retrieval_consolidation | retrieval_consolidation | lambda_fact=0.007 | 0.3750 | 0.87 | not_improved |
| 0084 scheduled_query_plus_test | scheduled_query_plus_test | (0068 decay) | 0.3472 | 0.54 | same as baseline |
| 0085 retrieval_consolidation | retrieval_consolidation | jost_power=2.5 | 0.3903 | 0.91 | not_improved |
| 0086 retrieval_consolidation | retrieval_consolidation | sigmoid_k=50.0 | 0.3876 | 0.88 | not_improved |
| 0087 retrieval_consolidation | retrieval_consolidation | sigmoid_k=20.0 | 0.3862 | 0.89 | not_improved |
| 0088 random | random | (0068 decay) | 0.3449 | 0.57 | same as baseline |
| 0089 scheduled_query_all | scheduled_query_all | (0068 decay) | 0.3449 | 0.57 | same as baseline |
| 0090 retrieval_consolidation | retrieval_consolidation | activation_weight=0.5 | 0.3965 | 0.88 | rejected (CV=0.3443) |
| 0091 retrieval_consolidation | retrieval_consolidation | alpha=1.0 | 0.3995 | 0.90 | not_improved (noise) |
| 0092 retrieval_consolidation | retrieval_consolidation | alpha=2.0 | 0.3800 | 0.86 | not_improved |

## Self-Criticism (0078-0092)
- **retrieval_consolidation ceiling ~0.39**: All 16 param combos yield 0.38-0.40 fixed split
- **alpha tuning**: alpha=1.0 → 0.3995, alpha=2.0 → 0.3800 (both within ceiling)
- **CV (0076 mult)**: 0.3459 ± 9.0% — tied with baseline. The +13% fixed-split improvement from retrieval_consolidation doesn't transfer to CV.
- **Retrieval consolidation conclusion**: Policy-level change cannot break the decay-function ceiling. The testing effect works (correlation=0.90) but overall is capped at ~0.39 on fixed split.

## What To Avoid (Updated)
- All retrieval_consolidation param tweaks — ceiling at 0.39-0.40, no combination helps
- All policies (random, scheduled_query_all, scheduled_query_plus_test) — identical to baseline
- Any further micro-tuning of retrieval_consolidation params with Jost+sigmoid decay

## FINAL STATUS (as of exp_lme_0092)
92 experiments total. Search has converged:
- Best (CV-validated): exp_lme_0008 CV=0.3466 (scheduled_query, multiplicative)
- retrieval_consolidation ceiling: ~0.39-0.40 fixed split, CV tied at 0.3459
- 16 consecutive no-improvement experiments (0077-0092)
- Jost+sigmoid floor decay ceiling confirmed at ~0.345 CV

## Next Step Direction
**Requires going outside allowed surface:**
1. **Evaluator weight change** (user approved): Change to additive formula
2. **Alternative embedding cache**: Different embedding model
3. **Protocol change**: Different train/test split strategy

## Next Step Direction
The Jost+sigmoid floor decay function has been thoroughly explored (70+ experiments). The remaining levers:
1. **Evaluator weight change** (user approved): Additive formula massively changes rankings but is a policy/escalation decision
2. **Alternative embedding cache**: Try different embedding models

## Evaluator Change: Additive Formula (Implemented 2026-03-21)
**Change**: evaluator.py line 479 — `overall_score = 0.50 * retrieval_score + 0.50 * plausibility_score`

### Results with Additive Formula

| Exp | Policy | Decay/Lambda | Fixed Split | CV Mean | CV% |
|-----|--------|-------------|-------------|---------|-----|
| exp_lme_0068 | scheduled_query | sigmoid (009/040) | 0.4694 | 0.3409 | 38.0% |
| exp_lme_0077 | retrieval_consolidation | sigmoid (009/040) | 0.6560 | 0.3409 | 38.0% |
| exp_lme_0093 | retrieval_consolidation | sigmoid (009/040, alpha=1.0) | 0.6559 | 0.3382 | 38.5% |
| exp_lme_0094 | retrieval_consolidation | pure_exp | 0.2928 | — | failed |
| exp_lme_0095 | retrieval_consolidation | hyperbolic | 0.6197 | 0.3250 | 41.5% |
| exp_lme_0096 | retrieval_consolidation | sigmoid (006/025) | 0.6319 | — | — |
| exp_lme_0097 | retrieval_consolidation | sigmoid (012/060) | 0.6635 | 0.3434 | 37.2% |
| exp_lme_0098 | retrieval_consolidation | sigmoid (015/080) | 0.6744 | 0.3271 | 36.3% |
| exp_lme_0099 | scheduled_query | sigmoid (009/040) | 0.4694 | 0.3409 | 38.0% |

### Key Findings
- **CV mean is SPLIT-DETERMINED**: scheduled_query and retrieval_consolidation give IDENTICAL CV distributions (0.3409 ± 38.0%) with additive formula
- **Policy only affects fixed split**: retrieval_consolidation boosts fixed split by +0.187 but CV is identical
- **Hyperbolic floor failed**: CV=0.3250, worse than sigmoid
- **Pure exponential failed catastrophically**: 0.2928 — sigmoid floor is essential
- **Lambda tuning**: Higher lambda (012/060) gave best fixed split (0.6635) but CV unchanged
- **CV% not reducible**: All experiments show 36-42% variance, inherent to dataset split design
- **Fold dominance confirmed**: the train/test split determines CV variance, not the algorithm
- **Policy effect preserved**: retrieval_consolidation still shows 0.90 correlation vs 0.57 for scheduled_query

### Conclusion
The additive formula is a scoring convention change only. It amplifies the retrieval_consolidation testing effect but does NOT improve CV robustness. The 38% variance is inherent to the dataset split design, not correctable via decay function or reactivation policy.

**Best with additive formula**: exp_lme_0098 (retrieval_consolidation, highest fixed split=0.6744) — CV=0.3271 ± 36.3%
**Best CV mean**: exp_lme_0097 (CV=0.3434) — but all within noise of 0.3409

**Definitive finding**: CV is determined by train/test splits, not algorithm. Both policies converge to 0.3409 ± 38%.

## FINAL STATUS (as of exp_lme_0099)
99 experiments total. Additive formula confirmed as scoring convention change only.

## Remaining Escalation Paths
1. **Alternative embedding cache**: Different embedding model (changes the data, not the algorithm)
2. **Protocol change**: Different train/test split strategy (addresses the 38% variance at its source)