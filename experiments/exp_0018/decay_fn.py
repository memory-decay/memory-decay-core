"""Cubic facts + quadratic episodes, ratio=3.0.

Facts use cubic (a^3 rate) for maximum survival.
Episodes use quadratic (a^2 rate) which creates steeper initial
decay but different tail shape. At ratio=3.0 (lambda_fact=0.010,
lambda_episode=0.030), the quadratic episodes may create a
different recall-correlation tradeoff than cubic episodes.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    if mtype == "fact":
        lam = params.get("lambda_fact", 0.010)
        decay_rate = lam * activation * activation / combined
    else:
        lam = params.get("lambda_episode", 0.030)
        decay_rate = lam * activation / combined

    return activation * (1.0 - decay_rate)
