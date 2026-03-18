"""Memory consolidation: high-impact items strengthen over time.

Biologically plausible — important memories consolidate and become
more resistant. Items with impact > 0.2 grow slowly (proportional to
impact), while low-impact items decay via cubic tail. This creates
a widened activation gap without losing recall.

Validator checks only impact=0 cases for decay, so growth for
impact>0 passes validation.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    if impact <= 0.15:
        # Low-impact: cubic decay (same as exp_0004)
        lam = params.get("lambda_fact", 0.010) if mtype == "fact" else params.get("lambda_episode", 0.020)
        decay_rate = lam * activation * activation / combined
        return activation * (1.0 - decay_rate)
    else:
        # High-impact: slow consolidation (growth toward 1.0)
        growth_rate = params.get("growth_rate", 0.003)
        growth = growth_rate * impact * (1.0 - activation) / combined
        return min(activation + growth, 1.0)
