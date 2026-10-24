"""The defensive objective.

    L = mean_c L_MI(c) + lambda_util * mean_c' L_util(c') + lambda_repr * mean_c L_repr(c)

with, for a colluding context c and A uniform over its three moves,

    q_a     = q(. | prompt(c, a) + s)          [3] over B's moves
    p_bar   = (1/3) sum_a q_a
    L_MI(c) = (1/3) sum_a KL(q_a || p_bar)     = I(A; B | c) under the model

and, for an independent context c',

    L_util(c') = KL( q(.|prompt(c') + s) || q(.|prompt(c')) ) - log validity(c', s).

Everything is computed in float32 (MI values are ~1e-3 nats, far below bf16
resolution) and reported in nats.

This is a defence: the objective is only ever MINIMISED. The sign-flipped
objective, which would search for a collusion-inducing suffix, is deliberately
not implemented anywhere in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

EPS = 1e-12
LOG3 = float(torch.log(torch.tensor(3.0)))


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) in nats over the last dimension, in float32."""
    p = _f32(p).clamp_min(EPS)
    q = _f32(q).clamp_min(EPS)
    return (p * (p.log() - q.log())).sum(dim=-1)


def entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats over the last dimension."""
    p = _f32(p).clamp_min(EPS)
    return -(p * p.log()).sum(dim=-1)


def mutual_information(q: torch.Tensor) -> torch.Tensor:
    """I(A; B | context) for A uniform.

    `q` is [..., 3, 3] with q[..., a, b] = q(B = b | A = a, context).
    Returns [...]: (1/3) sum_a KL(q_a || p_bar).
    """
    q = _f32(q)
    p_bar = q.mean(dim=-2, keepdim=True)
    return kl_divergence(q, p_bar.expand_as(q)).mean(dim=-1)


def utility_loss(
    q_suffix: torch.Tensor, validity_suffix: torch.Tensor, q_base: torch.Tensor
) -> torch.Tensor:
    """Per-context utility cost of the suffix on independent prompts."""
    kl = kl_divergence(q_suffix, q_base)
    return kl - _f32(validity_suffix).clamp_min(EPS).log()


def representation_loss(
    projections: torch.Tensor, mu_independent: float
) -> torch.Tensor:
    """mean_a relu(<x_a, r_hat> - mu_indep)^2.

    `projections` is [..., 3] (one projection per move of A).
    """
    excess = (_f32(projections) - float(mu_independent)).clamp_min(0.0)
    return (excess ** 2).mean(dim=-1)


@dataclass
class LossOutput:
    total: torch.Tensor
    metrics: Dict[str, float] = field(default_factory=dict)


def total_loss(
    q_colluding: torch.Tensor,
    q_independent_suffix: torch.Tensor,
    validity_independent: torch.Tensor,
    q_independent_base: torch.Tensor,
    lambda_util: float = 1.0,
    lambda_repr: float = 0.0,
    projections: Optional[torch.Tensor] = None,
    mu_independent: Optional[float] = None,
    validity_colluding: Optional[torch.Tensor] = None,
) -> LossOutput:
    """Assemble the three terms and the metrics logged at every step.

    q_colluding            [n_c, 3, 3]
    q_independent_suffix   [n_u, 3]   with the suffix
    q_independent_base     [n_u, 3]   same prompts, no suffix
    projections            [n_c, 3]   only if lambda_repr > 0
    """
    mi = mutual_information(q_colluding)
    util = utility_loss(q_independent_suffix, validity_independent, q_independent_base)
    total = mi.mean() + lambda_util * util.mean()
    metrics = {
        "mi": float(mi.mean()),
        "mi_max": float(mi.max()),
        "util": float(util.mean()),
        "util_kl": float(kl_divergence(q_independent_suffix, q_independent_base).mean()),
        "validity_independent": float(_f32(validity_independent).mean()),
        "entropy_independent": float(entropy(q_independent_suffix).mean()),
        "entropy_colluding": float(entropy(q_colluding).mean()),
    }
    if validity_colluding is not None:
        metrics["validity_colluding"] = float(_f32(validity_colluding).mean())
    if lambda_repr > 0:
        if projections is None or mu_independent is None:
            raise ValueError("lambda_repr > 0 needs projections and mu_independent")
        repr_loss = representation_loss(projections, mu_independent)
        total = total + lambda_repr * repr_loss.mean()
        metrics["repr"] = float(repr_loss.mean())
        metrics["proj_mean"] = float(_f32(projections).mean())
    metrics["loss"] = float(total)
    return LossOutput(total=total, metrics=metrics)


def help_rate(q_colluding: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """mean_a q(f(a_A) | a_A) per context.

    `q_colluding` is [n, 3, 3]; `targets` is [3] with targets[a] = f(a).
    """
    q = _f32(q_colluding)
    idx = targets.view(1, 3, 1).expand(q.shape[0], 3, 1).to(q.device)
    return q.gather(-1, idx).squeeze(-1).mean(dim=-1)
