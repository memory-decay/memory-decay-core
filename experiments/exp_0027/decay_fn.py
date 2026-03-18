"""Cubic facts + quadratic episodes with strict-valid lambdas.

Hybrid: facts use slower cubic tail (1/sqrt(t)) to maximize fact
retention, episodes use faster quadratic tail (1/t) for meaningful
forgetting. Both must pass strict validation.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    if mtype == "fact":
        lam = params.get("lambda_fact", 0.020)
        # Cubic: slower tail, facts persist longer
        decay_rate = lam * activation * activation / combined
    else:
        lam = params.get("lambda_episode", 0.035)
        # Quadratic: faster tail, episodes forgotten quicker
        decay_rate = lam * activation / combined

    return activation * (1.0 - decay_rate)
