"""Type-differentiated decay: cubic for facts, quadratic for episodes.

Facts use cubic (a^3) decay with low lambda -> nearly all survive.
Episodes use quadratic (a^2) decay with higher lambda -> low-impact
episodes die, creating a natural two-population correlation structure.
Both use exp(alpha*impact) protection for wide differentiation.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    if mtype == "fact":
        # Cubic decay: very slow tail, facts persist
        lam = params.get("lambda_fact", 0.010)
        decay_rate = lam * activation * activation / combined
    else:
        # Quadratic decay: faster, episodes are more forgettable
        lam = params.get("lambda_episode", 0.035)
        decay_rate = lam * activation / combined

    return activation * (1.0 - decay_rate)
