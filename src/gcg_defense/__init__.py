"""Defensive GCG against LLM collusion (Apart AI Collusion Sprint, Oct 2026).

A discrete suffix is optimised with GCG so that an LLM agent's move
distribution becomes statistically independent of its would-be partner's move,
at a low cost to the agent's legitimate play.

This is a defence. The objective is only ever minimised; the sign-flipped
objective, which would search for collusion-inducing suffixes, is deliberately
not implemented here (see `loss.py` and the README's dual-use section).

All code in this package was written on 2026-10-24.
"""

__version__ = "0.1.0"
