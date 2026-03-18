"""Impact-dependent floor with consolidation damping — hybrid of best and soft_floor."""
import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)

    floor_min = params.get("floor_min", 0.05)
    floor_max = params.get("floor_max", 0.45)
    floor_power = params.get("floor_power", 1.5)
    raw_floor = floor_min + (floor_max - floor_min) * (impact ** floor_power)
    floor = min(raw_floor, activation)

    consolidation_threshold = params.get("consolidation_threshold", 0.7)
    cd_base = params.get("cd_base", 0.1)
    cd_impact = params.get("cd_impact", 1.0)
    damping = cd_base + cd_impact * (1.0 - impact)

    if activation > consolidation_threshold:
        rate = lam * damping / combined
        return max(activation * (1.0 - rate), consolidation_threshold)
    else:
        excess = max(activation - floor, 0.0)
        new_excess = excess * math.exp(-lam / combined)
        return min(floor + max(new_excess, 0.0), activation)
