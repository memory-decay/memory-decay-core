"""Gompertz floor: bi-linear approach with gentler transition than sigmoid.

Gompertz function: floor = floor_max * exp(-beta * exp(-gamma * importance))
- At low importance: floor approaches floor_max * (1 - beta*exp(-gamma*importance)) ≈ slow decay
- At high importance: floor approaches 0 (but clamped to 0.01 minimum)
- The transition region is broader than sigmoid, potentially stabilizing fold variance

Key difference from sigmoid: asymmetric transition, slower initial decline.
Hypothesis: Gompertz's gentler, asymmetric transition may produce more stable
activation-recall correlation across CV folds compared to sigmoid's sharp switch.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # Importance from impact + stability
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    # Gompertz floor parameters
    floor_max = params.get("floor_max", 0.55)
    gomp_beta = params.get("gompertz_beta", 0.3)    # displacement (when decay starts)
    gomp_gamma = params.get("gompertz_gamma", 8.0)  # rate of decay
    floor_min = params.get("floor_min", 0.02)        # absolute floor

    # Gompertz: slower at high importance (resists forgetting)
    # exp(-beta * exp(-gamma * importance)) where importance in [0,1]
    gomp_arg = gomp_beta * math.exp(-gomp_gamma * importance)
    gomp_value = math.exp(-gomp_arg)

    floor = floor_max * gomp_value
    floor = max(floor, floor_min)
    floor = min(floor, activation)

    # Jost decay (same as exp_0315)
    lambda_fact = params.get("lambda_fact", 0.008)
    lambda_episode = params.get("lambda_episode", 0.035)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    effective_lambda = base_lambda / retention

    excess = max(activation - floor, 0.0)
    jost_power = params.get("jost_power", 4.0)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
