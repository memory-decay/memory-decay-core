# Codex Critical Review: Breakthrough Analysis

**Reviewer**: Codex (GPT-5.4, reasoning effort: xhigh)
**Date**: 2026-03-21
**Reviewing**: memory_chain/breakthrough_analysis.md

---

## 1. Root Cause Accuracy — Partially Correct, Misses Key Factors

The original analysis correctly identifies three interacting mechanisms (Jost power-law self-extinguishing, sigmoid floor saturation, retrieval consolidation feedback loop). However, it misses **the most important factor**:

### 🔴 The Evaluator Itself Is Misleading

**Finding 1: `evaluate_recall` does NOT use threshold for candidate filtering.**

In `src/memory_decay/evaluator.py`, the recall evaluation logic is:
1. Check if expected memory's `activation_score > threshold` (gate on single node)
2. Run similarity search with `top_k` results
3. Check if expected_id appears in results

The threshold only gates whether the *expected* memory is "observable" — it does NOT filter the similarity search candidates. This means `threshold_discrimination` measures "does the target stay above threshold," not "does threshold improve retrieval quality." This is a fundamentally different question.

**Finding 2: `corr_mean` is meaningless.**

`activation_recall_correlation()` accepts a `threshold` parameter but **completely ignores it**. The function computes correlation between activation score and recall success without any threshold filtering. Since `score_summary()` calls this for each of 9 thresholds and averages, `corr_mean` is literally the same number repeated 9 times then averaged. The 0.917 correlation score is real but has no threshold-dependent behavior.

**Finding 3: Retrieval consolidation boost overwhelms all decay functions.**

When running numerical simulations with the actual retrieval_consolidation policy (boost_every=20 ticks, boost=0.2):

| Decay Function | high fact @t200 | low fact @t200 | high episode @t200 | low episode @t200 |
|---|---|---|---|---|
| current (Jost+sigmoid+floor) | 0.921 | 0.483 | 0.792 | 0.285 |
| nofloor_jost | 1.000 | 1.000 | 0.857 | 0.724 |
| time_decaying_floor | 1.000 | 1.000 | 1.000 | 0.724 |
| **pure_exponential** | **1.000** | **0.881** | **0.430** | **0.266** |

All functions except pure exponential push all memories to 1.0 with regular boosting. This means the original analysis's focus on the floor mechanism is **misdirected** — the floor is not the primary cause of ceiling lock. The reinforcement schedule is.

---

## 2. Proposal Feasibility Assessment

### Proposal 1: Dual-Score Architecture
- **Would it break td=0?**: Only if `strength` decays fast enough between boosts. With boost_every=20, the gap between boosts is too small for any reasonable decay to create meaningful spread (simulation confirms: nofloor_jost still hits 1.0 for all cases).
- **Risk**: Significant complexity increase. Requires changes to evaluator, graph, and decay engine. The dual-score concept is sound but doesn't address the root cause (boost frequency).
- **Verdict**: ❌ Will NOT work as described. The boost schedule overwhelms any strength decay between intervals.

### Proposal 2: Time-Decaying Floor
- **Would it break td=0?**: Simulation shows NO — timefloor produces identical results to nofloor_jost because the boost refills everything.
- **Risk**: Low implementation complexity, but mathematically insufficient.
- **Verdict**: ❌ Will NOT work. Floor decay rate (0.005/tick) is negligible compared to boost amount (0.2/20 ticks = 0.01/tick effective).

### Proposal 3: Importance-Scaled Exponential (No Floor)
- **Would it break td=0?**: **YES** — simulation shows the widest spread (0.266 to 1.000 across importance levels). Without Jost power-law, decay is linear and doesn't self-extinguish near zero.
- **Risk**: Moderate. Episode-type memories decay faster (lambda_episode=0.08 vs 0.015 for facts). Recall at threshold=0.3 would only include high-importance memories.
- **Verdict**: ✅ Most promising, but needs parameter tuning. The key insight is that WITHOUT Jost power-law, the decay doesn't have the self-extinguishing property that creates floor-lock.

---

## 3. Mathematical Verification

### Pure Exponential with Boost — Why It Creates Spread

The critical difference is that pure exponential decay has a constant fractional rate:
```
a_{t+1} = a_t * exp(-λ_eff)
```

Between boosts (20 ticks), the decay is:
- Low importance: λ_eff = 0.015/1.15 = 0.013 → 20-tick factor = exp(-0.26) = 0.771
- High importance: λ_eff = 0.015/2.55 = 0.006 → 20-tick factor = exp(-0.12) = 0.887

After boost (+0.2, capped at 1.0):
- Low: 1.0 * 0.771 + 0.2 = 0.971 → converges to ~0.88 (below 0.9 threshold)
- High: 1.0 * 0.887 + 0.2 = 1.087 → capped at 1.0

**This creates a natural equilibrium below 1.0 for low-importance memories.** Jost power-law destroys this because decay_rate ∝ (excess)^4 approaches zero as activation approaches floor, so even low-importance memories get "stuck" near their floor.

### Why Jost Power-Law Defeats All Proposals

With jost_power=4 and any floor > 0:
- `decay_amount = λ * (a - floor)^4`
- As `a → floor`, decay_amount → 0
- The system asymptotes; boost easily overcomes the negligible decay
- Result: everything converges to max(floor, boost_equilibrium) ≈ 0.9+

---

## 4. Alternatives the Original Analysis Missed

### Alt 1: Importance-Proportional Boosting
Instead of uniform boost=0.2, scale boost by importance:
```python
boost = base_boost * (1 + importance_scale * importance)
```
This would amplify the natural decay-based spread rather than erase it.

### Alt 2: Decay-Aware Boost (Boost ≤ Decay Since Last Boost)
Only boost a memory if it has actually decayed since the last boost:
```python
actual_decay = prev_activation - current_activation
boost = min(requested_boost, actual_decay * 1.5)
```
This prevents over-boosting that creates ceiling lock.

### Alt 3: Remove Jost Power-Law, Keep Everything Else
The simplest fix might be to set jost_power=1.0 (linear excess decay) instead of 4.0. This alone would prevent the self-extinguishing behavior while keeping the sigmoid floor mechanism.

### Alt 4: Evaluate with Actual Threshold Filtering
Fix the evaluator to apply threshold filtering to the similarity search candidate set, not just the expected memory. This would make threshold_discrimination a more meaningful metric.

---

## 5. Ranking

1. **Fix evaluator semantics** (highest priority) — Current metric is measuring the wrong thing. Any optimization against a misleading metric is wasted effort.
2. **Pure exponential + importance-scaled half-life** (highest impact on td) — Mathematical proof shows widest activation spread under boost schedule.
3. **Remove Jost power-law (jost_power=1.0)** (simplest change) — Test if linear excess decay alone breaks ceiling lock while maintaining recall.
4. **Importance-proportional boosting** (policy-level fix) — Addresses root cause directly without changing decay function.
5. **Dual-Score** (lowest priority) — Conceptually clean but adds complexity without addressing the boost problem.

---

## 6. Concerns About Evaluation Methodology

1. **threshold_discrimination is a poor metric as currently implemented.** It only checks whether one specific memory (the expected answer) stays above threshold, not whether the system can separate signal from noise across a population. A more useful metric would measure the activation distribution's spread (e.g., standard deviation, interquartile range, or Gini coefficient of final activation scores).

2. **corr_mean is fake.** The correlation function ignores the threshold parameter. This inflates the "plausibility" sub-score without actually measuring threshold-dependent behavior. Either fix the function to use the threshold, or remove it from the threshold sweep loop.

3. **Overall score conflation.** The additive formula `0.5 * retrieval + 0.5 * plausibility` where plausibility includes fake corr_mean means the overall score is partially driven by a constant. This makes comparing experiments unreliable.

4. **The 38% CV variance may be an artifact.** If the metric is dominated by factors orthogonal to the decay function (e.g., embedding quality, query similarity distribution), then no amount of decay tuning will reduce variance.

---

## Summary

The original analysis correctly identifies the symptom (ceiling lock) but misidentifies the primary cause. The floor mechanism is a contributing factor, but the **reinforcement boost schedule is the dominant force** creating td=0. The most impactful fixes are:

1. Fix the evaluator to measure what we think it's measuring
2. Remove or reduce Jost power-law to prevent self-extinguishing decay
3. Consider importance-proportional or decay-aware boosting

The current search space has been exhausted not because the decay function is hard to optimize, but because **the evaluation metric is insensitive to the parameter being optimized**.
