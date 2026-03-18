"""Hybrid exponential/reciprocal with type diff."""
import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    floor_min = params.get("floor_min", 0.25)
    floor_max = params.get("floor_max", 0.60)
    floor_power = params.get("floor_power", 0.7)
    raw_floor = floor_min + (floor_max - floor_min) * (impact ** floor_power)
    floor = min(raw_floor, activation)

    base_approach = params.get("base_approach", 0.005)
    approach_impact = params.get("approach_impact", 6.0)
    type_fact_factor = params.get("type_fact_factor", 0.7)
    type_factor = type_fact_factor if mtype == "fact" else 1.0

    approach_rate = base_approach * type_factor * (1.0 + approach_impact * (1.0 - impact))

    excess = max(activation - floor, 0.0)

    blend_threshold = params.get("blend_threshold", 0.5)
    if activation > blend_threshold:
        # Exponential for smooth high-activation decay
        new_excess = excess * math.exp(-approach_rate)
    else:
        # Reciprocal for power-law tail near floor
        recip_rate = approach_rate * params.get("recip_scale", 3.0)
        new_excess = excess / (1.0 + recip_rate)

    return min(floor + max(new_excess, 0.0), activation)
