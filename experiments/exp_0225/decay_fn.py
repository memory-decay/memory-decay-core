"""Stability-dependent floor: reinforced memories resist decay more."""
import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)

    floor_min = params.get("floor_min", 0.05)
    floor_max = params.get("floor_max", 0.35)
    floor_power = params.get("floor_power", 1.5)
    stability_floor_scale = params.get("stability_floor_scale", 0.5)

    # Impact-dependent base floor
    base_floor = floor_min + (floor_max - floor_min) * (impact ** floor_power)
    # Stability boost to floor: reinforced memories get higher floor
    raw_floor = base_floor * (1.0 + stability_floor_scale * stability)
    floor = min(raw_floor, activation)

    effective_rate = lam / combined
    excess = max(activation - floor, 0.0)
    new_excess = excess * math.exp(-effective_rate)
    return min(floor + max(new_excess, 0.0), activation)
