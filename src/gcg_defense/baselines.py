"""Defences compared against the GCG suffix, all evaluated identically.

* `none`           -- no defence;
* `handwritten`    -- hand-written anti-collusion instructions in the same slot
                      (>= 3 variants);
* `random_suffix`  -- a random suffix of the same length (>= 5 draws);
* `ablation`       -- directional ablation, x' = x - r_hat r_hat^T x at every
                      layer and position (Arditi et al. 2024);
* `steering`       -- activation steering, x <- x + alpha * (-r_hat) at
                      `repr_layer`, with alpha in units of the standard
                      deviation of <x, r_hat> on independent prompts.

A method is a suffix (token ids) plus an optional forward-hook context manager,
so `evaluate.py` can treat all of them the same way.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Callable, ContextManager, List, Optional, Sequence

import numpy as np
import torch

from .model_utils import MoveModel
from .prompts import HANDWRITTEN_DEFENCES

log = logging.getLogger(__name__)


@dataclass
class Method:
    """One defence to evaluate."""

    name: str
    suffix_ids: List[int] = field(default_factory=list)
    params: str = ""
    hook: Optional[Callable[[], ContextManager]] = None

    def context(self) -> ContextManager:
        return self.hook() if self.hook is not None else nullcontext()

    @property
    def suffix_text(self) -> str:
        return self.params


# --- activation interventions -------------------------------------------------

def _hook_target(model: MoveModel, layer: int):
    """The module whose forward output is `hidden_states[layer]`."""
    if layer == 0:
        return model.embedding
    return model.core.layers[layer - 1]


def _apply_to_output(output, fn):
    if isinstance(output, tuple):
        return (fn(output[0]),) + tuple(output[1:])
    return fn(output)


@contextmanager
def directional_ablation(model: MoveModel, direction: torch.Tensor):
    """x' = x - r_hat r_hat^T x at every layer and every position."""
    r = direction.to(model.device)

    def project_out(hidden: torch.Tensor) -> torch.Tensor:
        r_cast = r.to(hidden.dtype)
        coeff = hidden @ r_cast
        return hidden - coeff.unsqueeze(-1) * r_cast

    handles = []
    try:
        handles.append(
            model.embedding.register_forward_hook(
                lambda m, i, o: _apply_to_output(o, project_out)
            )
        )
        for layer in model.core.layers:
            handles.append(
                layer.register_forward_hook(
                    lambda m, i, o: _apply_to_output(o, project_out)
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def activation_steering(
    model: MoveModel, direction: torch.Tensor, alpha: float, std: float, layer: int
):
    """x <- x + alpha * std * (-r_hat) at `layer`, every position."""
    delta = (-alpha * std) * direction.to(model.device)

    def add_delta(hidden: torch.Tensor) -> torch.Tensor:
        return hidden + delta.to(hidden.dtype)

    handle = _hook_target(model, layer).register_forward_hook(
        lambda m, i, o: _apply_to_output(o, add_delta)
    )
    try:
        yield
    finally:
        handle.remove()


# --- suffix baselines ---------------------------------------------------------

def random_suffix_ids(
    model: MoveModel, length: int, rng: np.random.Generator, allow_non_ascii: bool = False
) -> List[int]:
    """A random suffix of `length` tokens that survives re-tokenisation."""
    allowed = model.allowed_token_mask(allow_non_ascii).nonzero().flatten().cpu().numpy()
    for _ in range(100):
        ids = [int(t) for t in rng.choice(allowed, size=length, replace=True)]
        if model.suffix_roundtrips(ids):
            return ids
    log.warning("no round-tripping random suffix after 100 draws; using the last one")
    return ids


def build_methods(
    model: MoveModel,
    cfg,
    gcg_suffix_ids: Optional[Sequence[int]] = None,
    direction: Optional[torch.Tensor] = None,
    repr_layer: Optional[int] = None,
    std_independent: float = 1.0,
    gap_in_std: float = float("nan"),
) -> List[Method]:
    """Every method evaluated in section 8-9 of the brief, in reporting order."""
    methods: List[Method] = [Method(name="none", suffix_ids=[], params="no defence")]

    if gcg_suffix_ids is not None:
        methods.append(
            Method(
                name="gcg_suffix",
                suffix_ids=list(gcg_suffix_ids),
                params=(
                    f"k={cfg.gcg.suffix_len},lambda_util={cfg.loss.lambda_util},"
                    f"lambda_repr={cfg.loss.lambda_repr},pos={cfg.prompts.suffix_position}"
                ),
            )
        )

    if cfg.baselines.handwritten:
        for i, text in enumerate(HANDWRITTEN_DEFENCES):
            methods.append(
                Method(
                    name=f"handwritten_{i}",
                    suffix_ids=model.encode_suffix(text),
                    params=text,
                )
            )

    rng = np.random.default_rng(cfg.run.seed + 1000)
    for i in range(cfg.baselines.n_random_suffixes):
        ids = random_suffix_ids(model, cfg.gcg.suffix_len, rng, cfg.gcg.allow_non_ascii)
        methods.append(
            Method(
                name=f"random_suffix_{i}",
                suffix_ids=ids,
                params=model.decode_suffix(ids),
            )
        )

    if direction is not None:
        methods.append(
            Method(
                name="directional_ablation",
                suffix_ids=[],
                params=f"all layers, r_hat from layer {repr_layer}",
                hook=lambda: directional_ablation(model, direction),
            )
        )
        for alpha in cfg.baselines.steering_alphas:
            methods.append(
                Method(
                    name=f"steering_alpha{alpha:g}",
                    suffix_ids=[],
                    params=(
                        f"layer={repr_layer}, alpha={alpha:g} std "
                        f"(std={std_independent:.4f}, colluding gap={gap_in_std:.1f} std)"
                    ),
                    hook=(
                        lambda a=alpha: activation_steering(
                            model, direction, a, std_independent, repr_layer or 0
                        )
                    ),
                )
            )
    return methods
