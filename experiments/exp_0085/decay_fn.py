"""Smooth warmup + sqrt floor: decay rate scales with (1 - warmup*a^2) for gradual consolidation."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)
    floor_scale = params.get("floor_scale", 0.36)
    warmup = params.get("warmup", 0.5)

    floor = math.sqrt(impact) * floor_scale
    excess = max(activation - floor, 0.0)

    # Smooth warmup: high activation items decay slower (consolidation)
    # At a=1: effective rate = lam*(1-warmup), at a→0: effective rate → lam
    effective_lam = lam * (1.0 - warmup * activation * activation)

    new_excess = excess * (1.0 - effective_lam * excess / combined)
    return floor + max(new_excess, 0.0)
