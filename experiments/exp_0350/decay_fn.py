"""Power-mean floor with p=2.0 (emphasizes maximum of impact and stability).

Opposite of failed p=0.5 (bottleneck). p=2 gives more influence to the MAXIMUM
of (impact, stability). A memory with (impact=0.9, stability=0.1) gets
importance ≈ 0.64 (vs 0.70 with linear) — slightly lower but not catastrophic.
A memory with (impact=0.5, stability=0.5) gets 0.50 (vs 0.575) — noticeably lower.

Key effect: memories with moderate EITHER impact OR stability get a lower floor
than the linear blend would give. This is the opposite goal from the p=0.5
bottleneck attempt. It tests whether the "one-sided" memories should get
less floor protection (potentially helping precision by letting them decay faster).

This is mathematically equivalent to: importance = sqrt((imp² + stab²)/2),
which is the quadratic mean (RMS) of impact and stability.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)

    # Power-mean floor: p=2.0 (quadratic mean / RMS — emphasizes maximum)
    p = params.get("importance_p", 2.0)
    importance_raw = ((impact ** p) + (stability ** p)) / 2.0
    if importance_raw <= 0:
        importance = 0.0
    else:
        importance = importance_raw ** (1.0 / p)

    floor_max = params.get("floor_max", 0.55)
    sigmoid_k = params.get("sigmoid_k", 20.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.25)

    z = sigmoid_k * (importance - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        sigmoid = ez / (1.0 + ez)

    floor = floor_max * sigmoid
    floor = min(floor, activation)

    lambda_fact = params.get("lambda_fact", 0.015)
    lambda_episode = params.get("lambda_episode", 0.040)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    effective_lambda = base_lambda / retention

    excess = max(activation - floor, 0.0)
    jost_power = params.get("jost_power", 4.0)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
