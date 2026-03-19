"""Separated impact/stability roles: impact -> decay rate, stability -> floor only.

Hypothesis: Currently, impact and stability are combined into a single importance
value that controls both the decay rate AND the floor position. This means
high-stability/low-impact memories have low importance and get low floors,
preventing them from building up activation even via repeated reinforcement.

Separated design:
- Impact drives decay rate (high impact = slow decay via retention)
- Stability drives the floor (high stability = high floor, protects against erasure)
- This allows frequently-reactivated low-impact memories to accumulate retention
  without being penalized by a low floor
- The floor_min ensures memories never completely erase

Key insight: This is NOT equivalent to just reweighting alpha/rho because
the mathematical coupling is fundamentally different. With the current formula,
high stability with low impact still gives low importance = low floor. With
separation, high stability always gives a high floor regardless of impact.
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # --- Decay rate: governed by impact (high impact = slow decay) ---
    lambda_fact = params.get("lambda_fact", 0.008)
    lambda_episode = params.get("lambda_episode", 0.035)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    alpha = params.get("alpha", 1.5)
    # Retention from impact only (not stability here)
    retention = max(1.0 + alpha * impact, 1.0)
    effective_lambda = base_lambda / retention

    # --- Floor: governed by stability (high stability = high floor) ---
    floor_max = params.get("floor_max", 0.60)
    sigmoid_k = params.get("sigmoid_k", 20.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.25)
    floor_min = params.get("floor_min", 0.02)

    # Apply sigmoid to stability only for floor determination
    z = sigmoid_k * (stability - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        sigmoid = ez / (1.0 + ez)

    floor = floor_max * sigmoid
    floor = max(floor, floor_min)
    floor = min(floor, activation)

    # --- Jost decay with excess above floor ---
    excess = max(activation - floor, 0.0)
    jost_power = params.get("jost_power", 4.0)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
