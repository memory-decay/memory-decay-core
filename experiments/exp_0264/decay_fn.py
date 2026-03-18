"""Consolidation + reciprocal floor approach."""
import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)

    floor_min = params.get("floor_min", 0.25)
    floor_max = params.get("floor_max", 0.60)
    floor_power = params.get("floor_power", 0.7)
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
        # Reciprocal approach to floor
        base_approach = params.get("base_approach", 0.010)
        approach_impact = params.get("approach_impact", 5.0)
        type_fact_factor = params.get("type_fact_factor", 0.7)
        type_factor = type_fact_factor if mtype == "fact" else 1.0
        approach_rate = base_approach * type_factor * (1.0 + approach_impact * (1.0 - impact))

        excess = max(activation - floor, 0.0)
        new_excess = excess / (1.0 + approach_rate)
        return min(floor + new_excess, activation)
