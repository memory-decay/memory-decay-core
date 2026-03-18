"""Cubic decay, ratio=3.0, alpha=1.5 (reduced impact protection)."""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.010) if mtype == "fact" else params.get("lambda_episode", 0.030)
    decay_rate = lam * activation * activation / combined
    return activation * (1.0 - decay_rate)
