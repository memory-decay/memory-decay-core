"""Activation-dependent rate scaling for smoother convergence."""
import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)

    floor_min = params.get("floor_min", 0.05)
    floor_max = params.get("floor_max", 0.35)
    floor_power = params.get("floor_power", 1.5)
    raw_floor = floor_min + (floor_max - floor_min) * (impact ** floor_power)
    floor = min(raw_floor, activation)

    # Rate scales with activation: faster decay at high activation, slower near floor
    rate = lam * activation / combined
    excess = max(activation - floor, 0.0)
    new_excess = excess * math.exp(-rate)
    return min(floor + max(new_excess, 0.0), activation)
