"""Power-law decay toward impact-dependent floor."""
import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.015) if mtype == "fact" else params.get("lambda_episode", 0.04)

    floor_min = params.get("floor_min", 0.05)
    floor_max = params.get("floor_max", 0.35)
    floor_power = params.get("floor_power", 1.5)
    raw_floor = floor_min + (floor_max - floor_min) * (impact ** floor_power)
    floor = min(raw_floor, activation)

    effective_rate = lam / combined
    excess = max(activation - floor, 0.0)
    # Power-law: 1/(1+r) instead of exp(-r)
    new_excess = excess / (1.0 + effective_rate)
    return min(floor + new_excess, activation)
