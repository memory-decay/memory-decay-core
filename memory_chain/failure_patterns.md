# Memory Chain - Failure Pattern Analysis

## Overview

This document provides a systematic analysis of why the auto-research search converged at exp_0315 (Jost power = 4.0) after 46 experiments. It catalogs which structural changes improved performance and which systematically failed, providing context for understanding the theoretical ceiling of the current search surface and what directions remain viable for future exploration.

## What Worked

### Jost-plus-sigmoid (exp_0315, overall=0.2228)

- **Mechanism**: Jost Law decay with power=4.0 produces a steep, super-linear activation-dependent decay curve. The sigmoid floor at mid=0.25 creates a sharp importance-activation correlation that preserves high-importance memories while enabling rapid decay of low-importance ones.
- **Why it worked**: The super-linear Jost curvature (excess^4) provides strong separation between high and low activation memories. The sigmoid floor acts as a threshold that prevents low-importance memories from collapsing below retrievable plausibility while allowing high-importance memories to maintain elevated activation. This coupling is essential because it allows the importance signal to modulate the effective decay rate in a physically-motivated way (biological systems show activation-dependent decay rates).
- **Key parameters**: jost_power=4.0, sigmoid_mid=0.25, floor_max=0.55, recall_mean=0.402, plausibility=0.6511

### Association Boost (exp_0338, overall=0.2259 fixed-split)

- **Mechanism**: assoc_boost=2.0 applies a multiplicative factor to spreading activation during retrieval, amplifying the indirect activation contribution from episodic-to-fact associations.
- **Why it worked**: Spreading activation retrieves facts through episodic association chains. Boosting this mechanism improved retrieval_score to 0.2403 (vs exp_0315's 0.2351) by making indirectly-activated facts more accessible.
- **Caveat**: CV evaluation showed instability across folds (std=0.012 for exp_0315 vs higher variance for exp_0338). The fixed-split improvement did not generalize robustly, suggesting assoc_boost overfits to the specific split structure.

### Jost Power Trajectory (exp_0310 to exp_0315)

The Jost power parameter showed consistent improvement across its progression:

| Experiment | Jost Power | Overall | Retrieval | Plausibility |
|------------|------------|---------|-----------|---------------|
| exp_0310   | 2.0        | 0.1745  | 0.1834    | 0.6782        |
| exp_0311   | 2.0+floor  | 0.1802  | 0.1894    | 0.6750        |
| exp_0312   | 2.5        | 0.2086  | 0.2201    | 0.6513        |
| exp_0313   | 3.0        | 0.2115  | 0.2230    | 0.6573        |
| exp_0315   | 4.0        | 0.2228  | 0.2351    | 0.6511        |

The monotonic improvement suggests power=4.0 is near the optimal point within the Jost family, as further increases would likely cause excessive decay of even high-importance memories.

### Early Reinforcement Tuning (exp_0301 to exp_0308)

The initial improvements came from tuning reinforcement parameters rather than decay mechanics:

- **exp_0301** (0.1528): Introduced Jost Law decay + sigmoid floor + higher reinforcement_gain_assoc
- **exp_0305** (0.1584): Higher reinforcement (direct=0.35, assoc=0.20, cap=1.5) + slower fact decay
- **exp_0306** (0.1617): Stronger reinforcement (direct=0.40, assoc=0.30, cap=2.0) + lambda_fact=0.010
- **exp_0308** (0.1653): Slower decay, higher floor_max=0.60, stability_decay=0.005

These establish that reinforcement tuning and decay parameterization are complementary axes: the Jost power then provided a step-change improvement by directly modeling activation-dependent decay rates.

## What Failed

### Gompertz Floor (exp_0339, overall=0.2152)

- **Hypothesis**: Replace the sigmoid floor with a Gompertz function (asymmetric, approaching floor asymptotically) to better model biological forgetting curves.
- **Failure mode**: Plausibility collapsed to 0.5026 (vs 0.6511 for exp_0315). The sharp sigmoid transition is essential for maintaining the importance-activation correlation. Gompertz's gradual asymptotic approach to the floor caused all memories to converge toward the same low plausibility, destroying retrieval discrimination.
- **Root cause**: The sigmoid's sharp transition at mid=0.25 creates a threshold effect that Gompertz lacks. Without this threshold, low-importance memories do not drop below the retrieval threshold quickly enough, while high-importance memories cannot maintain elevated plausibility against the Gompertz floor.

### Decoupled Impact/Stability (exp_0340, overall=0.1506)

- **Hypothesis**: Separate the decay rate (impact) from the floor (stability) into independent parameters, allowing independent tuning of how fast memories decay versus how stable they become.
- **Failure mode**: Recall collapsed to 0.28 (vs 0.40 for exp_0315), the largest drop of any structural variant. Retrieval_score fell to 0.1608.
- **Root cause**: The coupling between impact and stability in the original formulation is not a modeling artifact but a structural necessity. In the Jost-plus-sigmoid model, importance modulates the effective decay rate, and the floor is reached at rates that depend on that modulation. Separating these roles breaks the importance-activation correlation that drives retrieval.

### Bounded Impact-Stability Interaction (exp_0341, overall=0.1989)

- **Hypothesis**: Add a bounded interaction term to prevent over-coupling of impact and stability within the importance model.
- **Failure mode**: Over-protected strong memories, reducing recall spread. Plausibility dropped to 0.5856.
- **Root cause**: The interaction bound prevented the model from fully exploiting the importance signal. Strong memories should be able to maintain very high activation under strong importance, but the bounded term capped this effect.

### Memory-Type-Specific Jost Curvature (exp_0342, overall=0.1871)

- **Hypothesis**: Apply different Jost power values to different memory types (episodic vs fact) to capture type-specific decay behavior.
- **Failure mode**: Type split hurt late-run recall more than it helped separation. Plausibility dropped to 0.5711.
- **Root cause**: The global Jost curvature provides a unified decay framework. Splitting by memory type fragmented the importance signal and prevented the super-linear effect from operating consistently across the memory store.

### Piecewise Jost Near Floor (exp_0343, overall=0.1989)

- **Hypothesis**: Apply a softer Jost curvature near the floor (where memories approach baseline) to prevent premature plausibility collapse.
- **Failure mode**: Late-run behavior converged to the same weaker regime as exp_0341. Plausibility matched exp_0341 at 0.5856.
- **Root cause**: The piecewise modification disrupted the globally-consistent curvature that makes power=4.0 effective. Near-floor behavior is better handled by the sigmoid floor than by modifying the Jost function itself.

### Dual-Sigmoid Floor (exp_0344, overall=0.1916)

- **Hypothesis**: Replace the single sigmoid floor with two sigmoids (one for activation-driven decay, one for importance-driven stabilization) to better model floor behavior.
- **Failure mode**: Improved some mid-run retrieval but could not preserve final 200-tick score. Plausibility dropped to 0.5485.
- **Root cause**: The dual-sigmoid complexity reduced the sharp threshold effect that is essential to the single sigmoid's performance. The additional parameters introduced underfitting in some regions and overfitting in others.

### Nonlinear Retention Weighting (exp_0345, overall=0.1992)

- **Hypothesis**: Apply nonlinear weighting to retention updates to emphasize high-importance memories more strongly.
- **Failure mode**: Slightly improved selectivity over exp_0341/0343 but still below exp_0315. Plausibility 0.5870.
- **Root cause**: The nonlinear weighting interacted poorly with the already-optimized importance mechanism, diluting the signal rather than amplifying it.

### Other Low-Scoring Variants

- **exp_0302** (0.1054): sigmoid_mid 0.25 to 0.40, lambda_episode 0.060 — too aggressive, recall dropped
- **exp_0303** (0.1251): sigmoid_mid 0.32, lambda_episode 0.050 — still worse than exp_0301
- **exp_0304** (0.1256): prune_factor 4.0 for low-importance — kills correct answers too
- **exp_0307** (0.1294): bi-exponential (fast+slow) — worse than Jost, too much slow-component for low-importance
- **exp_0309** (0.1474): jost_power=1.2 — too linear, insufficient activation separation
- **exp_0321** (0.1991): stability-enhanced importance + quadratic retention — over-weighted stability
- **exp_0324** (0.1394): hyperbolic decay + acceleration — Jost excess^4 still superior
- **exp_0328** (0.1820): activation_weight=1.0 — baseline without importance modulation
- **exp_0329** (0.1677): activation_weight=1.5 — insufficient modulation range

## Structural Diversity Analysis

The 46 experiments explored the following decay function families:

| Family | Variants | Best Score | Verdict |
|--------|----------|------------|---------|
| Jost Law (power 1.2-4.0) | 7 | 0.2228 | Best performing, converged at power=4.0 |
| Gompertz | 1 | 0.2152 | Failed — plausibility collapse |
| Bi-exponential | 1 | 0.1294 | Failed — slow component dominant |
| Hyperbolic | 1 | 0.1394 | Failed — inferior to Jost |
| Piecewise Jost | 1 | 0.1989 | Failed — inconsistent curvature |
| Dual-sigmoid | 1 | 0.1916 | Failed — lost threshold sharpness |
| Sigmoid floor (baseline) | - | 0.1528 | Initial success, Jost improved it |

Key findings:
- **Jost Law is the only decay family that improved performance** beyond the initial sigmoid-plus-reinforcement baseline
- **No alternative decay function outperformed Jost** when substituted for it
- **Structural modifications to the Jost+sigmoid core consistently failed** once power reached 4.0
- **Coupling between importance and activation is essential** — attempts to decouple or bound it caused large drops

## Convergence Diagnosis

The search converged at exp_0315 (Jost power=4.0) because this point represents the optimal balance within the Jost-plus-sigmoid decay family. The theoretical ceiling of ~0.347 overall_score (estimated from the convergence summary analysis) is not reachable within this search surface because:

1. **The embedding ceiling**: exp_0312 (jost_power=2.5) already hit the embedding ceiling with recall=sim_recall=0.390, meaning further improvements in recall are bottlenecked by embedding quality rather than decay mechanics.

2. **Plausibility floor**: The sigmoid floor's plausibility is bounded by the activation function's maximum output. At floor_max=0.55 and sigmoid_mid=0.25, plausibility plateaus around 0.65 even for high-importance memories.

3. **Coupling constraint**: The importance-activation coupling is essential; attempts to modify it (exp_0340, exp_0341) caused catastrophic recall collapse.

4. **Structural ceiling**: The Jost-plus-sigmoid formulation defines the search surface. Any improvement beyond ~0.2228 requires expanding the surface itself (new embedding backend, different retrieval representation, or alternative evaluation protocol).

The 10 experiments marked "improved" represent marginal gains within the same paradigm. The 35 experiments marked "recorded" include both incremental tuning and structural failures. No structural innovation broke through the ceiling established by exp_0315.

## Implications for Next Experiments

The failure patterns suggest the following directions for future exploration:

### Viable Directions

1. **Embedding backend upgrade**: The embedding ceiling (recall=0.390) is the most binding constraint. A higher-quality embedding model would directly improve sim_recall and enable the plausibility-retrieval trade-off to be reoptimized.

2. **Retrieval representation**: The current retrieval uses a fixed similarity threshold. Alternative retrieval mechanisms (e.g., attention-based reranking, k-NN with learned k) could improve selectivity without requiring better embeddings.

3. **Evaluation protocol changes**: Different benchmark splits or datasets may reveal different optimization surfaces where the Jost+sigmoid core performs differently.

### Non-Viable Directions (per failure patterns)

1. **Alternative decay functions**: Gompertz, bi-exponential, hyperbolic, and piecewise variants all failed. The search surface is exhausted for these families.

2. **Decoupled impact/stability**: exp_0340 demonstrated this approach causes recall collapse. The coupling is structural.

3. **Multi-sigmoid architectures**: Dual-sigmoid and bounded interactions did not improve on the single sigmoid.

4. **Memory-type-specific decay**: Type splitting fragmented the importance signal without compensating benefits.

5. **Further Jost power tuning**: Power=4.0 appears optimal. The marginal gain from power=3.0 (0.2115) to power=4.0 (0.2228) was 0.011, and the search found no improvements beyond power=4.0.

### Recommended Next Move

To break the current ceiling, the search surface itself must change. The most productive single intervention would be upgrading the embedding backend, as this would lift the recall ceiling that is currently limiting overall_score even with optimal decay mechanics.
