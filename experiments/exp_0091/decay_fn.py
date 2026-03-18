"""Reciprocal (harmonic) excess decay with sqrt(impact) floor.

Instead of multiplicative: excess * (1 - rate)  [exponential-like]
Uses reciprocal:           excess / (1 + rate)   [1/t power-law tail]

The 1/t tail means items decay MUCH slower at low activation,
keeping more items above retrieval thresholds long-term.
"""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)
    floor_scale = params.get("floor_scale", 0.36)

    floor = math.sqrt(impact) * floor_scale
    excess = max(activation - floor, 0.0)

    # Reciprocal decay: excess / (1 + lam * excess / combined)
    # Asymptotically decays as 1/t instead of exp(-t)
    new_excess = excess / (1.0 + lam * excess / combined)
    return floor + new_excess
