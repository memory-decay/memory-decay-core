"""Pure damping-based discrimination (no exp(alpha*impact) in combined)."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    rho = params.get("stability_weight", 0.8)
    combined = 1.0 + rho * stability
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)

    base_floor = params.get("base_floor", 0.79)
    impact_floor = params.get("impact_floor_scale", 0.01)
    floor = (base_floor + math.sqrt(impact) * impact_floor) * min(activation / 0.1, 1.0) if activation > 0 else 0.0

    consolidation_threshold = params.get("consolidation_threshold", 0.7)
    cd_base = params.get("cd_base", 0.1)
    cd_impact = params.get("cd_impact", 0.8)
    damping = cd_base + cd_impact * (1.0 - impact)

    if activation > consolidation_threshold:
        rate = lam * damping / combined
        return activation * (1.0 - rate)
    else:
        excess = max(activation - floor, 0.0)
        new_excess = excess * (1.0 - lam * excess / combined)
        return floor + max(new_excess, 0.0)
