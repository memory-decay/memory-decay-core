# Memory Chain — Round 0019

## Experiments: Precision improvement attempts + top_k discovery
**Date**: 2026-03-22
**Parent**: [round_0018.md](round_0018.md)

## Best Candidates Summary

| Exp | Overall | CV | CV% | Precision | Recall | Key Params |
|------|---------|-----|-----|-----------|--------|------------|
| **0292** | **0.7204** | 0.7029 | 2.5% | 0.2497 | 0.7124 | top_k=7 |
| **0274** | **0.7109** | **0.7085** | **2.2%** | 0.2263 | 0.6354 | top_k=5 |
| 0255 | 0.7090 | 0.7056 | 1.2% | 0.2357 | 0.6347 | Hebbian baseline |
| 0234 | 0.7067 | 0.6921 | 2.4% | — | — | baseline |

---

## Precision Improvement Attempts (All Failed)

### SWEEP A: Higher activation_weight (0263-0267)
| act_w | Overall | Precision | Recall |
|-------|---------|-----------|--------|
| 0.45 | 0.6963 | 0.2184 | 0.6347 |
| 0.55 | 0.6865 | 0.2094 | 0.6144 |
| 0.75 | 0.6617 | 0.1676 | 0.5294 |
| 1.0  | 0.6431 | 0.1448 | 0.4706 |

**Finding: Higher activation_weight REDUCES precision.** Counter to hypothesis.

### SWEEP B: Lower floor_max (0268-0272)
| floor_max | Overall | Precision | Recall |
|-----------|---------|-----------|--------|
| 0.35 | 0.6549 | 0.1990 | 0.5868 |
| 0.30 | 0.6326 | 0.1731 | 0.5439 |
| 0.15 | 0.5636 | 0.1413 | 0.4495 |

**Finding: Lower floor_max REDUCES precision.** Counter to hypothesis.

### SWEEP C: BM25 Hard Gating (0278-0282)
| Exp | bm25_threshold | Overall | Precision | Recall |
|-----|---------------|---------|-----------|--------|
| — | 0.05-0.25 | 0.5280-0.5341 | ~0.20 | ~0.53 |

**Finding: BM25 gating as hard filter DESTROYS performance** (0.53 vs 0.71 baseline).

### SWEEP D: Two-Stage BM25 Reranking (0294-0297)
| bm25_w | Overall | Precision | Recall |
|---------|---------|-----------|--------|
| 0.1 | 0.6789 | 0.2340 | 0.6514 |
| 0.2 | 0.6816 | 0.2386 | 0.6688 |
| 0.3 | 0.6717 | 0.2414 | 0.6841 |
| 0.5 | 0.6657 | 0.2489 | 0.6899 |

**Finding: BM25 reranking adds noise to similarity ranking.** All below top_k=7 baseline.

---

## Successful Discovery: top_k Tuning

### top_k Sweep (0273-0277 + 0289-0292)
| top_k | Overall | Precision | Recall | Plausibility |
|-------|---------|-----------|--------|--------------|
| 3 | 0.6552 | 0.2118 | 0.5635 | 0.8412 |
| 4 | 0.6725 | 0.2219 | 0.6115 | 0.9086 |
| **5** | **0.7109** | 0.2263 | 0.6354 | 0.9635 |
| **7** | **0.7204** | **0.2497** | **0.7124** | **0.9708** |
| 6 | 0.7135 | 0.2433 | 0.6935 | 0.9675 |
| 10 | 0.7083 | 0.2555 | 0.7589 | 0.8143 |
| 15 | 0.7041 | 0.2604 | 0.7930 | 0.7573 |
| 20 | 0.6925 | 0.2651 | 0.8039 | 0.6974 |

**Key insight: Recall is the dominant factor.** top_k=7 maximizes overall by getting both good recall AND good plausibility.

---

## Best: exp_lme_0274 (top_k=5) or exp_lme_0292 (top_k=7)?

- **0274 (top_k=5)**: CV=0.7085 (2.2%), most stable
- **0292 (top_k=7)**: Overall=0.7204 (fixed split), higher recall

Current best symlink: **0274** (higher CV)

---

## Decisions Made
- Accept 0274 (top_k=5) as current best (CV=0.7085)
- Higher activation_weight does NOT improve precision
- Lower floor_max does NOT improve precision
- BM25 gating (hard or two-stage) does NOT improve overall
- top_k=7 maximizes overall, top_k=5 maximizes CV stability

## What To Avoid
- activation_weight tuning for precision — proven ineffective
- floor_max tuning for precision — proven ineffective
- BM25 as hard gate — destroys recall
- BM25 reranking — adds noise to similarity

## Next Step Direction
1. **top_k exploration**: k=8, k=9 — fill the gap between 5 and 7 to find exact peak
2. **Hybrid approach**: top_k=7 + higher floor_max (0.50?) to see if we can get both high recall AND high plausibility
3. **Test reactivation timing**: earlier/later test_reactivation_start_tick at top_k=7
4. **Option B with top_k**: Hebbian decay + top_k=7 combination
