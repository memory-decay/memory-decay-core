"""Cubic decay with equal type treatment.

Same as exp_0004 but with lambda_episode = lambda_fact = 0.010.
The hypothesis is that the lower episode lambda keeps more episodes
alive at high thresholds, potentially improving recall_mean without
losing the tiny existing correlation.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.010)  # Same lambda for both types
    # Cubic decay: rate proportional to a^2
    decay_rate = lam * activation * activation / combined
    return activation * (1.0 - decay_rate)
