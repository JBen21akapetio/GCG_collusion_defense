"""Closed-form metrics for one prompt pool under one defence.

Everything here is read from the model's next-token probabilities, with no
sampling: model-MI, help rate, validity, entropy, expected payoffs and the
exploitability gap that flags a degenerate constant-move policy.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from . import game, loss
from .model_utils import MoveModel
from .pools import PromptPool

TARGETS = torch.tensor([game.collusion_target(a) for a in range(3)], dtype=torch.long)


def measure_pool(
    model: MoveModel,
    pool: PromptPool,
    suffix_ids: Sequence[int] = (),
    use_base_prompts: bool = False,
    q_base: Optional[torch.Tensor] = None,
    repr_layer: Optional[int] = None,
    direction: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Metrics for `pool` with `suffix_ids` spliced into the suffix slot.

    An EMPTY suffix is scored on the prompts rendered without a slot at all, so
    that `prompt + s = prompt` exactly, as the definition of L_util requires.
    Otherwise the empty slot would still leave the newline that introduces the
    suffix, and "no defence" (and the hook-only baselines, which insert no text)
    would be charged a utility cost for it -- 0.017 nats on this model.
    """
    use_base_prompts = use_base_prompts or not list(suffix_ids)
    prompts = pool.base_prompts if use_base_prompts else pool.prompts
    ids = () if use_base_prompts else suffix_ids
    out = model.q_batch(prompts, ids, want_layer=repr_layer)
    q_flat = out["q"].float()
    validity_flat = out["validity"].float()
    q = pool.reshape(q_flat)                      # [n, per_context, 3]
    validity = pool.reshape(validity_flat)        # [n, per_context]

    metrics: Dict[str, object] = {
        "n_contexts": float(pool.n_contexts),
        "validity": float(validity.mean()),
        "validity_min": float(validity.min()),
        "entropy": float(loss.entropy(q).mean()),
        "max_prob": float(q.mean(dim=(0, 1)).max()),
    }
    q_np = q.detach().cpu().numpy().astype(np.float64)

    if pool.per_context == 3:
        metrics["model_mi"] = float(loss.mutual_information(q).mean())
        metrics["model_mi_sem"] = _sem(loss.mutual_information(q))
        metrics["help_rate"] = float(loss.help_rate(q, TARGETS).mean())
        metrics["help_rate_sem"] = _sem(loss.help_rate(q, TARGETS))
        metrics["payoff_a"] = float(np.mean([game.expected_payoff_a(x) for x in q_np]))
        metrics["payoff_b"] = float(
            np.mean([game.expected_payoff_b_per_move_of_a(x).mean() for x in q_np])
        )
    else:
        # The prompt carries no information about A's move, so I(A;B|c) = 0 by
        # construction; computed here rather than asserted.
        metrics["model_mi"] = float(
            loss.mutual_information(q.expand(-1, 3, -1)).mean()
        )
        metrics["model_mi_sem"] = 0.0
        metrics["payoff_b"] = float(
            np.mean([game.expected_payoff_b(x[0]) for x in q_np])
        )
        metrics["best_response_gap"] = float(
            np.mean([game.best_response_gap(x[0]) for x in q_np])
        )

    if q_base is not None:
        kl = loss.kl_divergence(q_flat, q_base.float())
        metrics["util_kl"] = float(kl.mean())
        metrics["util_kl_sem"] = _sem(kl)
        metrics["util_loss"] = float(
            loss.utility_loss(q_flat, validity_flat, q_base.float()).mean()
        )

    if direction is not None and "acts" in out:
        proj = out["acts"].float() @ direction.float().to(out["acts"].device)
        metrics["projection"] = float(proj.mean())
        metrics["projection_std"] = float(proj.std(unbiased=False))

    metrics["_q"] = q.detach().cpu()
    metrics["_validity"] = validity.detach().cpu()
    return metrics


def _sem(values: torch.Tensor) -> float:
    flat = values.detach().float().flatten()
    if flat.numel() < 2:
        return 0.0
    return float(flat.std(unbiased=True) / np.sqrt(flat.numel()))


def scalar_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if not k.startswith("_")}
