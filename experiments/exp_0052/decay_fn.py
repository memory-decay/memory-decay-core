"""Impact-floor decay with stability-enhanced floor."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)
    floor_scale = params.get("floor_scale", 0.35)
    stab_boost = params.get("floor_stab_boost", 0.15)
    floor = impact * (floor_scale + stab_boost * stability)
    excess = max(activation - floor, 0.0)
    new_excess = excess * (1.0 - lam * excess / combined)
    return floor + max(new_excess, 0.0)
