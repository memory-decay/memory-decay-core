# Breakthrough Analysis: Why threshold_discrimination = 0.0

## The Problem

All top experiments (exp_lme_0098 and neighbors) show identical recall (0.6667) across ALL thresholds 0.1-0.9. This means every memory that can be recalled has activation_score > 0.9 — the system cannot distinguish important from unimportant memories via threshold.

## Root Cause: Three Interacting Mechanisms Create a Ceiling Lock

### 1. Jost Power-Law Decay is Self-Extinguishing

The decay formula (line 48 of decay_fn.py):

```
decay_amount = effective_lambda * (excess ** jost_power)
```

With `jost_power = 4.0` and `excess = activation - floor`:

| activation | floor | excess | excess^4 | decay_amount (λ=0.005) |
|-----------|-------|--------|----------|----------------------|
| 1.00 | 0.45 | 0.55 | 0.0915 | 0.000458 |
| 0.90 | 0.45 | 0.45 | 0.0410 | 0.000205 |
| 0.80 | 0.45 | 0.35 | 0.0150 | 0.000075 |
| 0.60 | 0.45 | 0.15 | 0.0005 | 0.0000025 |

The 4th power makes decay exponentially slower as activation approaches the floor. A memory at 0.60 decays ~180x slower than one at 1.00. This creates an asymptotic approach — memories get "stuck" well above the floor.

### 2. Sigmoid Floor Saturates for Any Non-Trivial Impact

The floor calculation:

```python
importance = (impact * 1.5 + stability * 1.0) / 2.5
floor = 0.45 * sigmoid(30.0 * (importance - 0.32))
```

The sigmoid with k=30 is essentially a step function:

| impact | stability | importance | sigmoid | floor |
|--------|-----------|------------|---------|-------|
| 0.1 | 0.0 | 0.060 | 0.0004 | ~0.00 |
| 0.3 | 0.0 | 0.180 | 0.013 | 0.006 |
| 0.5 | 0.0 | 0.300 | 0.354 | 0.159 |
| 0.5 | 0.5 | 0.500 | 0.996 | 0.448 |
| 0.7 | 0.0 | 0.420 | 0.938 | 0.422 |
| 0.7 | 0.5 | 0.620 | 1.000 | 0.450 |

Any memory with importance > 0.4 (i.e., most memories with non-trivial impact) gets floor ≈ 0.45. Combined with the Jost power-law, these memories asymptote to ~0.5-0.6 and stay there indefinitely.

### 3. Retrieval Consolidation Creates a Positive Feedback Loop

This is the critical amplifier. From main.py:

```python
if reactivation_policy == "retrieval_consolidation":
    apply_retrieval_consolidation(t)
```

Every eval interval:
1. Query all test memories by similarity
2. If a memory appears in top-k results → boost its activation AND stability
3. Higher stability → higher importance → higher floor → slower decay
4. Higher activation → more likely to appear in top-k (via activation_weight) → more boosts

**The loop**: Memory is recalled → gets boosted → floor rises via stability → decays less → stays recallable → gets boosted again.

Since all test memories start at activation=1.0 and the Jost decay is too slow to drop them below the similarity-retrieval threshold before the first consolidation event, EVERY memory enters the positive feedback loop. Once in, it never exits.

### Why the Trade-Off Exists (td vs overall)

Experiments with high td (like exp_lme_0045, td=0.38) achieve it by:
- Using aggressive decay that kills many memories entirely
- This drops recall (fewer memories survive) and kills correlation (no signal)

The anti-correlation exists because the ONLY way to get td > 0 in this architecture is to kill memories — but killing memories destroys all other metrics. **The architecture has no mechanism to let memories decay to DIFFERENT stable levels based on importance.**

## Proposed Architecture Changes

### Proposal 1: Dual-Score Architecture (Strength + Consolidation)

**Mechanism**: Split `activation_score` into two independent scores:

- **strength** — decays freely with NO floor, used for threshold gating in `evaluate_recall`
- **consolidation** — floor-protected, used for retrieval ranking in `query_by_similarity`

```python
# In evaluate_recall (threshold check):
if node["strength_score"] < threshold:  # NOT activation_score
    continue

# In query_by_similarity (ranking):
sim = cosine_sim * (consolidation ** activation_weight)  # consolidation for ranking
```

**Why it works**: Strength decays at different rates for different memories (high-impact decays slower but DOES eventually decay). Threshold sweep now sees a distribution of strength values across 0.0-1.0, giving td > 0. But consolidation (floor-protected) keeps high-impact memories rankable in similarity search, maintaining high recall and correlation.

**Key**: The floor protects retrieval ranking quality, but NOT the threshold gate. Low-importance memories drop below high thresholds while high-importance ones remain — exactly what td measures.

### Proposal 2: Time-Decaying Floor

**Mechanism**: Make the floor itself decay over time, so old unreinforced memories eventually lose their floor protection:

```python
age = current_tick - created_tick
floor_decay_factor = math.exp(-floor_decay_rate * age)
effective_floor = floor * floor_decay_factor
```

With `floor_decay_rate = 0.005`:
- At tick 50: floor_decay_factor = 0.78, effective_floor = 0.35
- At tick 100: floor_decay_factor = 0.61, effective_floor = 0.27
- At tick 200: floor_decay_factor = 0.37, effective_floor = 0.17

**Why it works**: Low-impact memories lose floor protection faster (their floor starts lower, so when multiplied by the decay factor, it approaches zero sooner). High-impact memories maintain their floor longer but still eventually decay if not reinforced. This creates a natural spread of activation scores.

**Interaction with retrieval_consolidation**: Actively recalled memories get stability boosts that counteract the floor decay (since importance includes stability). Unreinforced memories do not. This creates a natural separation — frequently-needed memories stay high, rarely-needed ones drift down.

### Proposal 3: Importance-Scaled Decay Rate (Remove Floor Entirely)

**Mechanism**: Instead of a floor + Jost power-law, use a simple exponential decay with rate inversely proportional to importance, but with a LINEAR (not 4th-power) excess term:

```python
# Replace the entire floor+Jost mechanism with:
half_life = base_half_life * (1 + importance_scale * importance)
effective_lambda = math.log(2) / half_life
new_activation = activation * math.exp(-effective_lambda)
```

With `base_half_life = 30` (ticks) and `importance_scale = 10`:
- impact=0.1, stability=0: half_life = 30 * (1 + 10 * 0.06) = 48 ticks → at tick 200: activation ≈ 0.05
- impact=0.5, stability=0: half_life = 30 * (1 + 10 * 0.30) = 120 ticks → at tick 200: activation ≈ 0.31
- impact=0.7, stability=0.5: half_life = 30 * (1 + 10 * 0.62) = 216 ticks → at tick 200: activation ≈ 0.52
- impact=1.0, stability=0: half_life = 30 * (1 + 10 * 0.60) = 210 ticks → at tick 200: activation ≈ 0.52

**Why it works**: Without a floor, memories decay to genuinely different levels based on their importance. The retrieval_consolidation loop still operates but can only slow the decay, not halt it (no floor to asymptote toward). At threshold 0.3, only the top-importance memories survive; at threshold 0.1, medium-importance ones also survive. This gives td > 0.

**Risk**: Without any floor, even high-importance memories eventually reach zero. This could hurt recall at long timescales. Mitigation: retrieval_consolidation boosts counteract the decay for actively-used memories.

## Recommendation

**Proposal 1 (Dual-Score)** is the most promising because:

1. It directly decouples the two conflicting requirements (threshold discrimination needs spread; recall needs stability)
2. It requires minimal change to the evaluation pipeline (just which field is checked where)
3. It preserves the existing floor mechanism for retrieval quality while enabling threshold discrimination
4. The retrieval_consolidation loop can boost consolidation (for ranking) without inflating strength (for thresholds), breaking the positive feedback loop that causes ceiling lock

**Proposal 2 (Time-Decaying Floor)** is the simplest to implement and could be tried first as a quick test — it's a single-line change to the decay function.

**Proposal 3 (No Floor)** is the cleanest conceptually but highest risk — it throws away the floor mechanism that contributes to the 0.917 correlation score.

## Verification Strategy

For any proposal, the key test is:
1. Run threshold_sweep and check td > 0
2. Check that recall at threshold=0.3 stays > 0.5
3. Check that correlation stays > 0.7
4. If all three hold, the trade-off is broken
