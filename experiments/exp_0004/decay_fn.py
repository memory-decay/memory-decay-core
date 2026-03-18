"""Cubic decay with exponential impact protection.

Uses da/dt = -lambda * a^3 / combined, giving a(t) = 1/sqrt(1/a0^2 + 2*lambda*t/combined).
This 1/sqrt(t) tail is even slower than quadratic 1/t, keeping more items
above higher thresholds (0.4, 0.5). Combined with exp(alpha*impact) and
lower lambdas, this should push recall_mean toward its ceiling (~0.50).
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.010) if mtype == "fact" else params.get("lambda_episode", 0.020)
    # Cubic decay: rate proportional to a^2
    decay_rate = lam * activation * activation / combined
    return activation * (1.0 - decay_rate)
