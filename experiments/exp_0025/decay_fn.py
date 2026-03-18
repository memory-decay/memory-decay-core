"""Cubic decay with higher lambda to pass strict validation.

Cubic with lambda_fact=0.020 gives fact(impact=0) at tick 200 = 0.289 < 0.30.
Higher lambda than the gamed exp_0004 (0.010) forces actual forgetting
while the cubic 1/sqrt(t) tail still preserves high-impact items.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.020) if mtype == "fact" else params.get("lambda_episode", 0.060)
    decay_rate = lam * activation * activation / combined
    return activation * (1.0 - decay_rate)
