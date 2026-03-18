"""Cubic decay with optimized type ratio (1.5:1 instead of 2:1).

exp_0004 uses lambda_episode/lambda_fact = 2.0, creating correlation
but losing recall@0.5. exp_0010 uses ratio=1.0, losing all correlation.

This experiment tries ratio=1.5 — a middle ground that should preserve
more recall@0.5 while retaining most of the correlation signal from
the type-based decay rate difference.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.010) if mtype == "fact" else params.get("lambda_episode", 0.015)
    # Cubic decay: rate proportional to a^2
    decay_rate = lam * activation * activation / combined
    return activation * (1.0 - decay_rate)
