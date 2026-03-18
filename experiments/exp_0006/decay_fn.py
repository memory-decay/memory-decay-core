"""Equilibrium-based decay: converge toward impact-dependent target.

Instead of decaying toward zero, items converge toward an equilibrium
activation proportional to their impact. High-impact items stabilize
at high activation (eq ≈ 0.48), low-impact items at low activation
(eq ≈ 0.18). This creates wide activation spread with high correlation
while keeping most items above retrieval thresholds.

da/dt = -k * (a - eq)  ->  a(t) = eq + (a0 - eq) * exp(-k*t)
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation < 0.001:
        return activation * 0.95  # Near-zero stays near-zero

    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    impact_factor = math.exp(alpha * impact)
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.020) if mtype == "fact" else params.get("lambda_episode", 0.030)
    k = lam / combined

    # Equilibrium: impact-dependent target activation
    eq_base = params.get("eq_base", 0.18)
    eq_scale = params.get("eq_scale", 0.30)
    eq = eq_base + eq_scale * impact
    # Guard: equilibrium must be below current activation (ensure decay)
    eq = min(eq, activation * 0.995)

    return eq + (activation - eq) * (1.0 - k)
