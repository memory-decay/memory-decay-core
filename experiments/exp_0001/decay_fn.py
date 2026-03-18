"""Quadratic decay: power-law tail via activation-dependent rate.

da/dt = -lambda * a^2 / combined
Solution: a(t) = 1 / (1/a0 + lambda*t/combined) — a 1/t power law.
"""


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 0.8)
    rho = params.get("stability_weight", 0.8)
    impact_factor = 1.0 + alpha * impact
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.022)
    decay_rate = lam * activation / combined
    return activation * (1.0 - decay_rate)
