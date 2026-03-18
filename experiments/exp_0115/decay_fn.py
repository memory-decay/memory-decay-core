"""Two-phase decay with higher sqrt floor (0.45) and ultra-slow decay."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.008) if mtype == "fact" else params.get("lambda_episode", 0.024)
    floor_scale = params.get("floor_scale", 0.45)
    consolidation_threshold = params.get("consolidation_threshold", 0.7)
    consolidation_damping = params.get("consolidation_damping", 0.4)
    floor = math.sqrt(impact) * floor_scale

    if activation > consolidation_threshold:
        rate = lam * consolidation_damping / combined
        return activation * (1.0 - rate)
    else:
        excess = max(activation - floor, 0.0)
        new_excess = excess * (1.0 - lam * excess / combined)
        return floor + max(new_excess, 0.0)
