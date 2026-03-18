"""Use soft_floor_decay_step from decay.py."""

from memory_decay.decay import soft_floor_decay_step


def compute_decay(activation, impact, stability, mtype, params):
    lam = params.get("lambda_fact", 0.012) if mtype == "fact" else params.get("lambda_episode", 0.036)
    return soft_floor_decay_step(
        activation,
        impact,
        stability,
        lam=lam,
        alpha=params.get("alpha", 2.0),
        rho=params.get("stability_weight", 0.8),
        floor_min=params.get("floor_min", 0.05),
        floor_max=params.get("floor_max", 0.35),
        floor_power=params.get("floor_power", 2.0),
        gate_center=params.get("gate_center", 0.4),
        gate_width=params.get("gate_width", 0.08),
        consolidation_gain=params.get("consolidation_gain", 0.6),
        min_rate_scale=params.get("min_rate_scale", 0.1),
    )
