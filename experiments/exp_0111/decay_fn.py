"""Sigmoidal impact gating: sharp keep/forget bifurcation based on impact."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    k = params.get("sigmoid_steepness", 10.0)
    midpoint = params.get("sigmoid_midpoint", 0.4)
    lam_base = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)
    rho = params.get("stability_weight", 0.8)

    # Sigmoid gate: 0→1 as impact increases past midpoint
    gate = 1.0 / (1.0 + math.exp(-k * (impact - midpoint)))

    # High gate = very slow decay; low gate = fast decay
    # Effective rate ranges from lam_base*3 (low impact) to lam_base*0.1 (high impact)
    rate_multiplier = 0.1 + 2.9 * (1.0 - gate)
    effective_lam = lam_base * rate_multiplier

    # Stability protection
    protection = 1.0 + rho * stability
    rate = effective_lam / protection

    # Floor: gate-weighted so only high-impact items get a floor
    floor = gate * params.get("floor_scale", 0.35)

    excess = max(activation - floor, 0.0)
    new_excess = excess * (1.0 - rate)
    return floor + max(new_excess, 0.0)
