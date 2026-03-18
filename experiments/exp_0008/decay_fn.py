"""Quartic decay: ultra-slow 1/t^(1/3) tail.

da/dt = -lambda * a^4 / combined
Solution: a(t) = (3*lambda*t/combined + 1/a0^3)^(-1/3)

Even slower than cubic (1/sqrt(t)), keeping items at higher activation
levels. With a0=0.5, lambda=0.008: a(200) ≈ 0.44 (vs cubic 0.35).
This should push recall@0.5 closer to the 0.498 ceiling.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.008) if mtype == "fact" else params.get("lambda_episode", 0.015)
    # Quartic decay: rate proportional to a^3
    decay_rate = lam * activation * activation * activation / combined
    return activation * (1.0 - decay_rate)
