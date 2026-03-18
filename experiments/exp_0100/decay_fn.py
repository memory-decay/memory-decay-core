"""Impact-dependent lambda + sqrt floor: low-impact items get higher lambda
for faster forgetting, creating double impact differentiation."""
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 2.0)
    rho = params.get("stability_weight", 0.8)
    combined = max(math.exp(alpha * impact) * (1.0 + rho * stability), 1e-9)
    base_lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)
    floor_scale = params.get("floor_scale", 0.36)
    impact_lam_boost = params.get("impact_lam_boost", 2.0)

    # Low-impact items get higher effective lambda
    # impact=0: lam * (1 + boost) = 3x base
    # impact=1: lam * 1 = base
    effective_lam = base_lam * (1.0 + impact_lam_boost * (1.0 - impact))

    floor = math.sqrt(impact) * floor_scale
    excess = max(activation - floor, 0.0)
    new_excess = excess * (1.0 - effective_lam * excess / combined)
    return floor + max(new_excess, 0.0)
