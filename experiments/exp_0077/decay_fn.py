"""Two-phase decay: slow consolidation above threshold, quadratic floor decay below."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)
    floor_scale = params.get("floor_scale", 0.36)
    consolidation_threshold = params.get("consolidation_threshold", 0.6)
    consolidation_damping = params.get("consolidation_damping", 0.3)
    floor = math.sqrt(impact) * floor_scale

    if activation > consolidation_threshold:
        # Consolidation phase: damped linear decay, slower to let items stabilize
        rate = lam * consolidation_damping / combined
        return activation * (1.0 - rate)
    else:
        # Floor-approach phase: quadratic excess decay
        excess = max(activation - floor, 0.0)
        new_excess = excess * (1.0 - lam * excess / combined)
        return floor + max(new_excess, 0.0)
