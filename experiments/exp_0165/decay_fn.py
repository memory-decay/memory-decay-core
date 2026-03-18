"""Single-phase floor-approach with impact-dependent rate."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)

    base_floor = params.get("base_floor", 0.79)
    impact_floor = params.get("impact_floor_scale", 0.01)
    floor = (base_floor + math.sqrt(impact) * impact_floor) * min(activation / 0.1, 1.0) if activation > 0 else 0.0

    # Single phase: always approach floor
    excess = max(activation - floor, 0.0)
    rate = lam / combined
    new_excess = excess * (1.0 - rate)
    return floor + max(new_excess, 0.0)
