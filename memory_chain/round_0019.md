# Memory Chain — Round 0019

## Experiments: H1-H7 sweep + jost_power + Option B + lambda_ep fine-tune
**Date**: 2026-03-22
**Parent**: [round_0018.md](round_0018.md)

## All Results Summary

### Jost Power Sweep (0249-0254)
All at: lambda_fact=0.050, lambda_episode=0.200, act_w=0.35

| Exp | jost_power | Overall | Plausibility |
|-----|-----------|---------|--------------|
| 0250 | 2.5 | **0.7134** | 0.9750 |
| 0251 | 3.0 | 0.7027 | 0.9629 |
| 0252 | 3.5 | 0.6894 | 0.9572 |
| 0249 | 2.0 | 0.6877 | 0.8350 |
| 0253 | 4.0 | 0.6545 | 0.8790 |
| 0254 | 4.5 | 0.6484 | 0.8981 |

### Option B Alternative Mechanisms (0255-0257)
All at: lambda_fact=0.050, lambda_episode=0.200, act_w=0.35

| Exp | Mechanism | Overall | Plausibility | CV | CV% |
|-----|-----------|---------|--------------|-----|-----|
| **0255** | **Hebbian-decay** | **0.7090** | **0.9554** | **0.7056** | **1.2%** |
| 0257 | Adaptive threshold | 0.5875 | 0.5008 | — | fail |
| 0256 | Hyperbolic floor | 0.5727 | 0.8009 | — | fail |

### Lambda Episode Fine-Tune (0258-0262)
All at: lambda_fact=0.050, act_w=0.35

| Exp | lambda_ep | Overall | Plausibility |
|-----|-----------|---------|--------------|
| 0234 | 0.200 | 0.7067 | 0.9691 |
| 0258 | 0.170 | 0.7045 | 0.9745 |
| 0259 | 0.185 | 0.7038 | 0.9662 |
| 0260 | 0.195 | 0.7001 | 0.9674 |
| 0262 | 0.215 | 0.7005 | 0.9478 |
| 0261 | 0.205 | 0.6950 | 0.9550 |

---

## NEW BEST: exp_lme_0255 (Hebbian-decay)

**Key insight:** The Hebbian-decay mechanism — which modulates decay rate by distance-from-floor — achieves the best cross-validation score (CV=0.7056, 1.2%) despite slightly lower fixed-split than 0250. The mechanism is fundamentally different from Jost's Law and validates the distance-from-floor hypothesis.

| Metric | Value |
|--------|-------|
| Overall (fixed split) | 0.7090 |
| Retrieval | — |
| Plausibility | 0.9554 |
| CV mean | 0.7056 |
| CV std | 0.0088 |
| CV% | **1.2%** (extremely stable) |

**Params:** lambda_fact=0.050, lambda_episode=0.200, act_w=0.35, jost_power=3.0 + distance_scale modulation

## Decisions Made
- Accept exp_lme_0255 (Hebbian-decay) as new best
- Update `best` symlink to exp_lme_0255
- Hebbian-decay (distance-from-floor modulation) validated as best mechanism
- Hyperbolic floor and Adaptive threshold both FAIL — discard these directions
- Jost power 2.5 (0250) has best fixed-split but lower CV than 0255
- lambda_episode=0.200 confirmed optimal

## What To Avoid
- Hyperbolic floor mechanism — proven to fail
- Adaptive threshold mechanism — proven to fail
- Lambda_episode values away from 0.200 — no improvement
- High jost_power (>3.5) — degrades performance

## Next Step Direction
The decay mechanism search is maturing. Remaining open questions:
1. **Fine-tune Hebbian-decay distance_scale parameter** (0.3, 0.7, 1.0?) — the mechanism works but the coefficient matters
2. **Combine Hebbian + jost_power=2.5** — does the mechanism work even better with lower jost_power?
3. **Stability/reinforcement tuning** at Hebbian-decay optimal params
