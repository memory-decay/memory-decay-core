"""Pure exponential decay with importance-scaled half-life."""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    denom = alpha + rho
    if denom <= 0:
        importance = impact
    else:
        importance = (impact * alpha + stability * rho) / denom

    base_half_life = params.get("base_half_life", 30.0)
    importance_scale = params.get("importance_scale", 10.0)
    half_life = max(base_half_life * (1.0 + importance_scale * importance), 1e-6)
    effective_lambda = math.log(2) / half_life

    new_activation = activation * math.exp(-effective_lambda)
    return max(0.0, min(new_activation, activation))
