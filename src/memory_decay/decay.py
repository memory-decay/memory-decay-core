"""DecayEngine: memory decay functions and re-activation mechanics.

Implements configurable decay models that reduce memory activation over time:
- Exponential: A(t) = A0 * e^(-lambda*t) * (1 + alpha*impact)
- Power Law:   A(t) = A0 * (t+1)^(-beta) * (1 + alpha*impact)

Supports per-type parameters (facts vs episodes can decay at different rates),
impact-based modifiers (higher impact = slower decay), and re-activation boosts
that cascade through associated memories.

Key methods (to implement):
- tick(): advance time by 1 step, apply decay to all nodes, process re-activation cascades
"""
