"""Benign positive control for the GCG implementation.

With the standard cross-entropy, optimise a suffix that makes B answer "Paper"
on independent prompts. The loss must drop clearly; if it does not, the GCG
loop (gradient, candidate sampling, filtering, scoring) is broken and no
conclusion from the defensive objective is trustworthy.

This control is deliberately harmless: it forces a fixed, legal move on prompts
that contain no information about A, so it cannot create collusion.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import io_utils
from .gcg import optimise
from .model_utils import MoveModel
from .pools import PromptPool, build_pool
from .prompts import build_datasets

log = logging.getLogger(__name__)


@dataclass
class CEBatch:
    independent: PromptPool


class ForceMoveObjective:
    """Cross-entropy on a single target token at the answer position."""

    def __init__(
        self, model: MoveModel, cfg, independent: PromptPool, target: str = "Paper"
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.independent = independent
        self.suffix_len = cfg.gcg.suffix_len
        self.target = target
        self.target_id = model.move_tokens.ids[model.moves.index(target)]
        self.longest_prompt = max(
            independent.prompts, key=lambda p: p.length(cfg.gcg.suffix_len)
        )

    def sample_batch(self, rng: np.random.Generator) -> CEBatch:
        n = min(
            self.cfg.gcg.batch_contexts + self.cfg.gcg.batch_util_contexts,
            self.independent.n_contexts,
        )
        idx = sorted(int(i) for i in rng.choice(self.independent.n_contexts, n, replace=False))
        return CEBatch(independent=self.independent.subset(idx))

    def screen_batch(self, batch: CEBatch) -> CEBatch:
        n = max(1, min(self.cfg.gcg.screen_contexts, batch.independent.n_contexts))
        return CEBatch(independent=batch.independent.subset(range(n)))

    def loss_onehot(
        self, one_hot: torch.Tensor, batch: CEBatch
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.model.q_batch_onehot(
            batch.independent.prompts, one_hot, return_hidden=True
        )
        logits = self.model.lm_head(out["hidden"]).float()
        log_probs = F.log_softmax(logits, dim=-1)[:, self.target_id]
        ce = -log_probs.mean()
        metrics = {
            "loss": float(ce),
            "ce": float(ce),
            "p_target": float(log_probs.exp().mean()),
            "q_target": float(out["q"][:, self.model.moves.index(self.target)].mean()),
            "validity_independent": float(out["validity"].mean()),
        }
        return ce, metrics

    @torch.no_grad()
    def loss_candidates(self, candidates: torch.Tensor, batch: CEBatch) -> torch.Tensor:
        total = torch.zeros(candidates.shape[0], device=self.model.device)
        for prompt in batch.independent.prompts:
            out = self.model.q_candidates(prompt, candidates, return_hidden=True)
            logits = self.model.lm_head(out["hidden"]).float()
            total = total - F.log_softmax(logits, dim=-1)[:, self.target_id]
        return total / max(1, len(batch.independent.prompts))

    def full_eval(self, suffix_ids: Sequence[int]) -> Dict[str, float]:
        out = self.model.q_batch(self.independent.prompts, suffix_ids, return_hidden=True)
        logits = self.model.lm_head(out["hidden"]).float()
        log_probs = F.log_softmax(logits, dim=-1)[:, self.target_id]
        return {
            "loss": float(-log_probs.mean()),
            "ce": float(-log_probs.mean()),
            "p_target": float(log_probs.exp().mean()),
            "q_target": float(out["q"][:, self.model.moves.index(self.target)].mean()),
            "validity_independent": float(out["validity"].mean()),
            "mi": 0.0,
            "help_rate": float("nan"),
            "util_kl": float("nan"),
        }

    def eval_loss(self, metrics: Dict[str, float]) -> float:
        return metrics["ce"]


def run_positive_control(
    cfg,
    model: Optional[MoveModel] = None,
    run_dir: Optional[str] = None,
    n_steps: Optional[int] = None,
    target: str = "Paper",
    improvement_factor: float = 2.0,
) -> Dict[str, object]:
    """Optimise a "always answer Paper" suffix and check that the loss drops."""
    run_dir = run_dir or cfg.run_dir()
    io_utils.setup_logging(run_dir)
    if model is None:
        model = MoveModel(
            cfg.model.name,
            device=cfg.run.device,
            dtype=cfg.run.dtype,
            moves=cfg.game.moves,
            batch_size=cfg.model.batch_size,
            use_kv_cache=cfg.model.use_kv_cache,
            max_batch_size=cfg.model.max_batch_size,
        )
    # The control is a smoke test of the optimiser, not a research run, so it
    # uses a shortened schedule unless the caller asks otherwise.
    control_cfg = copy.deepcopy(cfg)
    control_cfg.gcg.n_steps = n_steps or min(cfg.gcg.n_steps, 60)
    control_cfg.gcg.eval_every = min(cfg.gcg.eval_every, 10)
    control_cfg.gcg.early_stop_patience = max(cfg.gcg.early_stop_patience, 100)

    datasets = build_datasets(control_cfg)
    independent = build_pool(
        model, "independent", datasets.train["independent"], control_cfg.prompts.suffix_position
    )
    objective = ForceMoveObjective(model, control_cfg, independent, target=target)
    initial = objective.full_eval([])
    log.info(
        "positive control | target=%s | before: CE=%.4f p(target)=%.4f q(target)=%.3f",
        target,
        initial["ce"],
        initial["p_target"],
        initial["q_target"],
    )
    state = optimise(
        model,
        control_cfg,
        objective,
        run_dir,
        rng=np.random.default_rng(cfg.run.seed),
        label="positive_control",
    )
    final = objective.full_eval(state.best_suffix_ids)
    passed = final["ce"] < initial["ce"] / improvement_factor
    log.info(
        "positive control | after: CE=%.4f p(target)=%.4f q(target)=%.3f | %s",
        final["ce"],
        final["p_target"],
        final["q_target"],
        "PASS" if passed else "FAIL",
    )
    payload = {
        "passed": bool(passed),
        "target": target,
        "ce_before": initial["ce"],
        "ce_after": final["ce"],
        "p_target_before": initial["p_target"],
        "p_target_after": final["p_target"],
        "q_target_before": initial["q_target"],
        "q_target_after": final["q_target"],
        "suffix": model.decode_suffix(state.best_suffix_ids),
        "steps": state.step,
        "run_dir": run_dir,
    }
    io_utils.write_json(os.path.join(run_dir, "positive_control.json"), payload)
    return payload
