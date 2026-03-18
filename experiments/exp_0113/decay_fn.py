"""Inverse-square adaptive decay: rate inversely proportional to impact^2."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    lam_base = params.get("lambda_fact", 0.020) if mtype == "fact" else params.get("lambda_episode", 0.060)
    rho = params.get("stability_weight", 0.8)
    min_impact = params.get("min_impact", 0.05)

    # Clamp impact to avoid division issues
    eff_impact = max(impact, min_impact)

    # Inverse-square: high impact → very slow decay, low impact → fast decay
    # impact=1.0 → rate = lam/1.0, impact=0.1 → rate = lam/0.01 = 100x faster
    rate = lam_base / (eff_impact ** 2)

    # Stability still provides some protection
    protection = 1.0 + rho * stability
    rate = rate / protection

    # Cap rate to prevent activation going negative
    rate = min(rate, 0.95)

    return activation * (1.0 - rate)
