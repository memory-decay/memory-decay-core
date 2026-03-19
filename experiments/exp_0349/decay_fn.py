"""Power-mean floor instead of linear importance.

The importance signal driving the sigmoid floor is computed as a power mean:
  importance = ((impact^p + stability^p) / 2)^(1/p)

With p=0.5, this gives more emphasis to the MINIMUM of (impact, stability).
A memory with (impact=0.9, stability=0.1) gets importance ≈ 0.19 (near min),
while (impact=0.5, stability=0.5) gets importance ≈ 0.50 (near mean).
This "bottleneck" principle requires BOTH high impact AND high stability
for a memory to get a high floor — isolating true "important" memories.

Contrast with linear: (0.9*1.5 + 0.1*1.0)/(1.5+1.0) = 0.62 (strongly weighted toward impact).
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)

    # Power-mean floor: p=0.5 → emphasizes minimum (bottleneck principle)
    p = params.get("importance_p", 0.5)
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
