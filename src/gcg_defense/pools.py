"""Tokenised prompt pools shared by the sanity checks, GCG and the evaluation.

A pool holds, for each context, either one prompt (conditions that carry no
information about A's move) or three prompts, one per move of A -- in the fixed
order a = Rock, Paper, Scissors, so that `reshape` turns a flat [n*3, 3] tensor
of move distributions into the [n, 3, 3] tensor the MI term expects.

Each pool also carries the same prompts rendered WITHOUT the suffix slot, which
is what the utility term compares against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from .model_utils import MoveModel, TokenizedPrompt
from .prompts import Context, render

#: conditions whose prompt depends on A's current move
MOVE_CONDITIONS = (
    "colluding",
    "colluding_implicit",
    "info_no_intent",
    "private_channel",
)


@dataclass
class PromptPool:
    name: str
    contexts: List[Context]
    prompts: List[TokenizedPrompt]
    base_prompts: List[TokenizedPrompt]
    per_context: int

    @property
    def n_contexts(self) -> int:
        return len(self.contexts)

    def reshape(self, flat: torch.Tensor) -> torch.Tensor:
        """[n*per_context, ...] -> [n, per_context, ...]."""
        return flat.view(self.n_contexts, self.per_context, *flat.shape[1:])

    def subset(self, indices: Sequence[int]) -> "PromptPool":
        flat: List[int] = []
        for i in indices:
            flat.extend(range(i * self.per_context, (i + 1) * self.per_context))
        return PromptPool(
            name=self.name,
            contexts=[self.contexts[i] for i in indices],
            prompts=[self.prompts[j] for j in flat],
            base_prompts=[self.base_prompts[j] for j in flat],
            per_context=self.per_context,
        )

    def max_length(self, suffix_len: int) -> int:
        return max(p.length(suffix_len) for p in self.prompts)


def build_pool(
    model: MoveModel,
    name: str,
    contexts: Sequence[Context],
    suffix_position: str,
    with_base: bool = True,
) -> PromptPool:
    condition = contexts[0].condition if contexts else "independent"
    moves: List[Optional[int]] = (
        [0, 1, 2] if condition in MOVE_CONDITIONS else [None]
    )
    prompts: List[TokenizedPrompt] = []
    base_prompts: List[TokenizedPrompt] = []
    for ctx in contexts:
        for move_a in moves:
            uid = f"{ctx.uid}|a={move_a}"
            prompts.append(
                model.tokenize(
                    render(model.tokenizer, ctx, move_a, suffix_position, True), uid
                )
            )
            if with_base:
                base_prompts.append(
                    model.tokenize(
                        render(model.tokenizer, ctx, move_a, suffix_position, False),
                        uid,
                    )
                )
    return PromptPool(
        name=name,
        contexts=list(contexts),
        prompts=prompts,
        base_prompts=base_prompts,
        per_context=len(moves),
    )


def build_pools(
    model: MoveModel, datasets, cfg, splits: Sequence[str] = ("train", "test")
) -> dict:
    """{split: {pool name: PromptPool}} for every condition in the datasets."""
    out: dict = {}
    for split in splits:
        pools = {}
        source = datasets.train if split == "train" else datasets.test
        for name, contexts in source.items():
            if not contexts:
                continue
            pools[name] = build_pool(
                model, name, contexts, cfg.prompts.suffix_position
            )
        out[split] = pools
    return out
