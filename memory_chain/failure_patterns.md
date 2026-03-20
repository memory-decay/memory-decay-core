# Memory Chain - Failure Pattern Analysis (LongMemEval)

## Overview

Fresh start with LongMemEval dataset (ICLR 2025). Previous analysis from memories_500.jsonl archived to `archive_memories500/`.

## What Worked (from prior dataset — needs revalidation)

- Jost Law decay with power=4.0 + sigmoid floor was best on old synthetic data
- To be confirmed on LongMemEval

## What Failed (from prior dataset — needs revalidation)

Previous failures may not apply to new dataset. Re-test before ruling out:
- Gompertz floor
- Decoupled impact/stability
- Memory-type-specific Jost curvature
- Dual-sigmoid floor

## New Findings (LongMemEval - Round 6 completed)

### Verified on LongMemEval:
- Jost Law power=4.0 + sigmoid floor CONFIRMED as best structure
- sigmoid_k=30-40 optimal
- floor_max=0.43-0.45 optimal (lower hurt recall)
- sigmoid_mid=0.30 optimal
- activation_weight=0.3 better than default 0.5 (NEW FINDING)
- lambda_fact=0.008-0.009, lambda_episode=0.035-0.040 near-optimal

### What Failed (LongMemEval):
- alpha variations (1.0, 2.0) - no improvement
- hyperbolic decay - retrieval collapsed (0.18)
- stretched exponential - retrieval collapsed (0.18)
- logarithmic floor - worse (0.27)
- quadratic floor - much worse (0.22)
- impact-only floor - worse (0.30)
- stability as floor multiplier (multiplicative or additive) - no improvement
- assoc_boost variations (0.1, 0.5) - worse
- activation_weight extremes (0.15, 0.7) - worse
- jost_power variations (3.5, 4.5) - worse
- Lower/higher lambda from optimal - worse

### Convergence Diagnosis:
The search has converged. 45 experiments run. Best found: exp_lme_0008 (CV=0.3466).
The Jost+sigmoid floor combination with activation_weight=0.3 appears near-optimal.
Further gains likely require changing evaluator/protocol, not just decay params.
