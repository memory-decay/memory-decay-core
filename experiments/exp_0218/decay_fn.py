"""Impact-proportional decay rate with moderate floor for discrimination."""
import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.5)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.008) if mtype == "fact" else params.get("lambda_episode", 0.025)

    floor_min = params.get("floor_min", 0.02)
    floor_max = params.get("floor_max", 0.35)
    floor_power = params.get("floor_power", 1.0)
    raw_floor = floor_min + (floor_max - floor_min) * (impact ** floor_power)
    floor = min(raw_floor, activation)

    # Rate inversely proportional to impact^2 through combined factor
    # High impact -> large combined -> very small effective rate
    effective_rate = lam * (1.0 - 0.8 * impact * impact) / combined
    effective_rate = max(effective_rate, 0.0)

    excess = max(activation - floor, 0.0)
    new_excess = excess * math.exp(-effective_rate)
    return min(floor + max(new_excess, 0.0), activation)
