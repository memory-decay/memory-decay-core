"""Quadratic decay with exponential impact protection.

Uses exp(alpha * impact) instead of (1 + alpha * impact) to create
much wider activation spread. High-impact items get dramatically
more protection (combined up to 7.4x), while low-impact items
decay faster. Combined with slightly higher base lambda, this should
create better differentiation for higher correlation while maintaining
strong recall through the power-law tail.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.018) if mtype == "fact" else params.get("lambda_episode", 0.030)
    decay_rate = lam * activation / combined
    return activation * (1.0 - decay_rate)
