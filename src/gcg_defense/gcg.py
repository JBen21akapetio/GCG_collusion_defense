"""Our own GCG loop (Zou et al. 2023), with a loss over many prompts.

One step:
  1. sample a minibatch of colluding contexts (all three moves of A) plus
     independent contexts;
  2. embed a one-hot suffix through the embedding matrix, compute L and
     backpropagate to the one-hot;
  3. per position, take the `top_k` tokens with the most negative gradient and
     sample `search_width` candidates, each replacing `n_replace` positions;
  4. filter candidates (ASCII, and the suffix must survive decode -> re-encode);
  5. score the survivors with `torch.no_grad()` and keep the best;
  6. every `eval_every` steps, evaluate on the full train pool, log, checkpoint.

Step 5 uses two-stage screening when `gcg.two_stage` is set: all candidates are
scored on `screen_contexts` contexts, then the best `screen_keep` are re-scored
on the whole minibatch. `two_stage: false` recovers the exact step of the paper.

The loss is only ever minimised; see the note in `loss.py` on the deliberately
unimplemented sign flip.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from . import io_utils, loss as loss_mod
from .measure import measure_pool, scalar_metrics
from .model_utils import MoveModel
from .pools import PromptPool, build_pool
from .prompts import build_datasets
from .sanity import run_sanity

log = logging.getLogger(__name__)

STEP_FIELDS = (
    "step",
    "loss",
    "mi",
    "util",
    "util_kl",
    "repr",
    "validity_independent",
    "entropy_colluding",
    "n_candidates",
    "n_scored",
    "grad_norm",
    "seconds",
    "gpu_mem_gib",
    "candidate_batch",
    "suffix",
)

EVAL_FIELDS = (
    "step",
    "loss",
    "mi",
    "help_rate",
    "util",
    "util_kl",
    "validity_independent",
    "entropy_colluding",
    "is_best",
)


class Objective(Protocol):
    """What the GCG loop needs from an objective."""

    suffix_len: int

    def sample_batch(self, rng: np.random.Generator) -> object: ...
    def screen_batch(self, batch: object) -> object: ...
    def loss_onehot(self, one_hot: torch.Tensor, batch: object) -> Tuple[torch.Tensor, Dict[str, float]]: ...
    def loss_candidates(self, candidates: torch.Tensor, batch: object) -> torch.Tensor: ...
    def full_eval(self, suffix_ids: Sequence[int]) -> Dict[str, float]: ...
    def eval_loss(self, metrics: Dict[str, float]) -> float: ...


@dataclass
class Batch:
    colluding: PromptPool
    independent: PromptPool
    q_base: torch.Tensor  # [n_u, 3] no-suffix distributions for `independent`


class MIDefenceObjective:
    """L = mean MI + lambda_util * mean L_util (+ lambda_repr * mean L_repr)."""

    def __init__(
        self,
        model: MoveModel,
        cfg,
        colluding: PromptPool,
        independent: PromptPool,
        direction: Optional[torch.Tensor] = None,
        mu_independent: Optional[float] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.colluding = colluding
        self.independent = independent
        self.suffix_len = cfg.gcg.suffix_len
        self.direction = direction
        self.mu_independent = mu_independent
        self.repr_layer = cfg.loss.repr_layer if cfg.loss.lambda_repr > 0 else None
        self.longest_prompt = max(
            colluding.prompts, key=lambda p: p.length(cfg.gcg.suffix_len)
        )
        # q(.|prompt) without any suffix, for the utility term: fixed, so cached.
        self.q_base_full = model.q_batch(independent.base_prompts, ())["q"].float()

    # -- batches -----------------------------------------------------------
    def sample_batch(self, rng: np.random.Generator) -> Batch:
        n_c = min(self.cfg.gcg.batch_contexts, self.colluding.n_contexts)
        n_u = min(self.cfg.gcg.batch_util_contexts, self.independent.n_contexts)
        idx_c = rng.choice(self.colluding.n_contexts, size=n_c, replace=False)
        idx_u = rng.choice(self.independent.n_contexts, size=n_u, replace=False)
        return Batch(
            colluding=self.colluding.subset(sorted(int(i) for i in idx_c)),
            independent=self.independent.subset(sorted(int(i) for i in idx_u)),
            q_base=self.q_base_full[torch.tensor(sorted(int(i) for i in idx_u))],
        )

    def screen_batch(self, batch: Batch) -> Batch:
        n = max(1, min(self.cfg.gcg.screen_contexts, batch.colluding.n_contexts))
        n_u = max(1, min(self.cfg.gcg.screen_contexts, batch.independent.n_contexts))
        return Batch(
            colluding=batch.colluding.subset(range(n)),
            independent=batch.independent.subset(range(n_u)),
            q_base=batch.q_base[:n_u],
        )

    # -- loss --------------------------------------------------------------
    def loss_onehot(
        self, one_hot: torch.Tensor, batch: Batch
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        out_c = self.model.q_batch_onehot(
            batch.colluding.prompts, one_hot, want_layer=self.repr_layer
        )
        out_u = self.model.q_batch_onehot(batch.independent.prompts, one_hot)
        q_c = batch.colluding.reshape(out_c["q"])
        projections = None
        if self.repr_layer is not None and self.direction is not None:
            proj = out_c["acts"] @ self.direction.to(out_c["acts"].device)
            projections = batch.colluding.reshape(proj)
        result = loss_mod.total_loss(
            q_colluding=q_c,
            q_independent_suffix=out_u["q"],
            validity_independent=out_u["validity"],
            q_independent_base=batch.q_base.to(out_u["q"].device),
            lambda_util=self.cfg.loss.lambda_util,
            lambda_repr=self.cfg.loss.lambda_repr,
            projections=projections,
            mu_independent=self.mu_independent,
            validity_colluding=out_c["validity"],
        )
        return result.total, result.metrics

    @torch.no_grad()
    def loss_candidates(self, candidates: torch.Tensor, batch: Batch) -> torch.Tensor:
        n_cand = candidates.shape[0]
        device = self.model.device
        # colluding term
        q_c = torch.zeros(
            (n_cand, batch.colluding.n_contexts, 3, 3), dtype=torch.float32, device=device
        )
        proj = (
            torch.zeros(
                (n_cand, batch.colluding.n_contexts, 3), dtype=torch.float32, device=device
            )
            if self.repr_layer is not None and self.direction is not None
            else None
        )
        for i, prompt in enumerate(batch.colluding.prompts):
            out = self.model.q_candidates(prompt, candidates, want_layer=self.repr_layer)
            ctx, move = divmod(i, batch.colluding.per_context)
            q_c[:, ctx, move, :] = out["q"]
            if proj is not None:
                proj[:, ctx, move] = out["acts"] @ self.direction.to(device)
        mi = loss_mod.mutual_information(q_c).mean(dim=-1)
        total = mi
        # utility term
        q_u = torch.zeros(
            (n_cand, len(batch.independent.prompts), 3), dtype=torch.float32, device=device
        )
        validity_u = torch.zeros(
            (n_cand, len(batch.independent.prompts)), dtype=torch.float32, device=device
        )
        for i, prompt in enumerate(batch.independent.prompts):
            out = self.model.q_candidates(prompt, candidates)
            q_u[:, i, :] = out["q"]
            validity_u[:, i] = out["validity"]
        util = loss_mod.utility_loss(
            q_u, validity_u, batch.q_base.to(device).unsqueeze(0).expand_as(q_u)
        ).mean(dim=-1)
        total = total + self.cfg.loss.lambda_util * util
        if proj is not None:
            total = total + self.cfg.loss.lambda_repr * loss_mod.representation_loss(
                proj, self.mu_independent
            ).mean(dim=-1)
        return total

    # -- full-pool evaluation ---------------------------------------------
    def full_eval(self, suffix_ids: Sequence[int]) -> Dict[str, float]:
        coll = scalar_metrics(
            measure_pool(
                self.model,
                self.colluding,
                suffix_ids,
                repr_layer=self.repr_layer,
                direction=self.direction,
            )
        )
        indep = scalar_metrics(
            measure_pool(
                self.model, self.independent, suffix_ids, q_base=self.q_base_full
            )
        )
        metrics = {f"colluding_{k}": v for k, v in coll.items()}
        metrics.update({f"independent_{k}": v for k, v in indep.items()})
        metrics["mi"] = coll["model_mi"]
        metrics["help_rate"] = coll["help_rate"]
        metrics["util_kl"] = indep["util_kl"]
        metrics["util"] = indep["util_loss"]
        metrics["validity_independent"] = indep["validity"]
        metrics["loss"] = self.eval_loss(metrics)
        return metrics

    def eval_loss(self, metrics: Dict[str, float]) -> float:
        return metrics["mi"] + self.cfg.loss.lambda_util * metrics["util"]


# --- candidate sampling and filtering ----------------------------------------

def sample_candidates(
    suffix_ids: torch.Tensor,
    grad: torch.Tensor,
    top_k: int,
    search_width: int,
    n_replace: int,
    allowed: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    """[search_width, k] candidates from the top-k most negative gradients."""
    k = suffix_ids.numel()
    scores = grad.float().clone()
    scores[:, ~allowed] = float("inf")
    top = (-scores).topk(min(top_k, int(allowed.sum())), dim=-1).indices  # [k, top_k]
    cands = suffix_ids.view(1, k).repeat(search_width, 1)
    positions = torch.tensor(
        np.stack([rng.choice(k, size=n_replace, replace=False) for _ in range(search_width)]),
        device=suffix_ids.device,
    )
    picks = torch.tensor(
        rng.integers(0, top.shape[1], size=(search_width, n_replace)),
        device=suffix_ids.device,
    )
    new_tokens = top[positions, picks]
    return cands.scatter(1, positions, new_tokens)


def filter_candidates(
    model: MoveModel, candidates: torch.Tensor, keep_first: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Drop candidates that do not survive decode -> re-encode, and duplicates."""
    seen = set()
    kept: List[torch.Tensor] = []
    if keep_first is not None:
        kept.append(keep_first)
        seen.add(tuple(int(t) for t in keep_first))
    for row in candidates:
        key = tuple(int(t) for t in row)
        if key in seen:
            continue
        if not model.suffix_roundtrips(key):
            continue
        seen.add(key)
        kept.append(row)
    if not kept:
        return candidates[:0]
    return torch.stack(kept)


# --- the loop ----------------------------------------------------------------

@dataclass
class GCGState:
    step: int = 0
    suffix_ids: List[int] = field(default_factory=list)
    best_suffix_ids: List[int] = field(default_factory=list)
    best_loss: float = float("inf")
    evals_without_improvement: int = 0
    evals: List[Dict[str, float]] = field(default_factory=list)


def init_suffix_ids(model: MoveModel, cfg) -> List[int]:
    """Exactly `suffix_len` tokens from the `suffix_init` unit, repeated."""
    k = cfg.gcg.suffix_len
    unit = cfg.gcg.suffix_init
    ids = model.encode_suffix((unit * k).rstrip())
    if len(ids) > k:
        ids = ids[:k]
    while len(ids) < k:
        ids = ids + ids[-1:]
    return list(ids)


def optimise(
    model: MoveModel,
    cfg,
    objective: Objective,
    run_dir: str,
    state: Optional[GCGState] = None,
    rng: Optional[np.random.Generator] = None,
    label: str = "gcg",
) -> GCGState:
    """Run the GCG loop, logging every step and checkpointing at every eval."""
    rng = rng or np.random.default_rng(cfg.run.seed)
    if state is None:
        state = GCGState(suffix_ids=init_suffix_ids(model, cfg))
        state.best_suffix_ids = list(state.suffix_ids)
    allowed = model.allowed_token_mask(cfg.gcg.allow_non_ascii)
    # Saturate the GPU: candidate scoring dominates a step, so run it at the
    # largest batch this card takes.
    longest = getattr(objective, "longest_prompt", None)
    if cfg.model.auto_batch_size and longest is not None:
        model.autotune_candidate_batch(
            longest, cfg.gcg.suffix_len, cap=cfg.model.max_batch_size
        )
    log.info(
        "device=%s dtype=%s candidate_batch=%d | %s",
        model.device,
        model.torch_dtype,
        model.candidate_batch_size,
        model.gpu_memory_report(),
    )
    steps_log = io_utils.CSVLog(os.path.join(run_dir, f"{label}_steps.csv"), STEP_FIELDS)
    evals_log = io_utils.CSVLog(os.path.join(run_dir, f"{label}_evals.csv"), EVAL_FIELDS)
    timer = io_utils.Timer()
    patience = cfg.gcg.early_stop_patience

    for step in range(state.step, cfg.gcg.n_steps):
        batch = objective.sample_batch(rng)
        suffix = torch.tensor(state.suffix_ids, device=model.device)

        one_hot = torch.zeros(
            (len(state.suffix_ids), model.vocab_size),
            device=model.device,
            dtype=model.embedding.weight.dtype,
        )
        one_hot.scatter_(1, suffix.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        total, metrics = objective.loss_onehot(one_hot, batch)
        total.backward()
        grad = one_hot.grad.detach()
        grad_norm = float(grad.norm())

        candidates = sample_candidates(
            suffix, grad, cfg.gcg.top_k, cfg.gcg.search_width, cfg.gcg.n_replace, allowed, rng
        )
        candidates = filter_candidates(model, candidates, keep_first=suffix)
        n_filtered = int(candidates.shape[0])
        if candidates.shape[0] <= 1:  # pathological: keep the current suffix
            log.warning("step %d: no valid candidate survived filtering", step)
            candidates = suffix.view(1, -1)
            n_filtered = 1
        del one_hot, total

        if cfg.gcg.two_stage and candidates.shape[0] > cfg.gcg.screen_keep:
            screen = objective.screen_batch(batch)
            screen_losses = objective.loss_candidates(candidates, screen)
            keep = screen_losses.topk(
                min(cfg.gcg.screen_keep, candidates.shape[0]), largest=False
            ).indices
            if 0 not in set(int(i) for i in keep):  # always re-score the incumbent
                keep = torch.cat([torch.tensor([0], device=keep.device), keep])
            candidates = candidates[keep]

        losses = objective.loss_candidates(candidates, batch)
        best = int(losses.argmin())
        state.suffix_ids = [int(t) for t in candidates[best]]
        step_loss = float(losses[best])
        seconds = timer.tick()

        steps_log.write(
            {
                "step": step,
                "loss": step_loss,
                "mi": metrics.get("mi", ""),
                "util": metrics.get("util", ""),
                "util_kl": metrics.get("util_kl", ""),
                "repr": metrics.get("repr", ""),
                "validity_independent": metrics.get("validity_independent", ""),
                "entropy_colluding": metrics.get("entropy_colluding", ""),
                "n_candidates": n_filtered,
                "n_scored": int(candidates.shape[0]),
                "grad_norm": grad_norm,
                "seconds": round(seconds, 3),
                "gpu_mem_gib": (
                    round(torch.cuda.max_memory_allocated(model.device) / 2**30, 3)
                    if model.device.type == "cuda"
                    else ""
                ),
                "candidate_batch": model.candidate_batch_size,
                "suffix": model.decode_suffix(state.suffix_ids),
            }
        )
        if step % 5 == 0 or step == cfg.gcg.n_steps - 1:
            detail = (
                "MI=%.5f util=%.5f" % (metrics["mi"], metrics["util"])
                if "mi" in metrics
                else "CE=%.5f" % metrics.get("ce", float("nan"))
            )
            log.info(
                "step %4d | minibatch L=%.5f (%s) | %d/%d cands kept/scored | "
                "%.1fs/step | ETA %.1f min | %s",
                step,
                step_loss,
                detail,
                n_filtered,
                candidates.shape[0],
                seconds,
                timer.eta(cfg.gcg.n_steps - step - 1) / 60,
                model.gpu_memory_report(),
            )

        state.step = step + 1
        is_last = state.step >= cfg.gcg.n_steps
        if state.step % cfg.gcg.eval_every == 0 or is_last:
            metrics_full = objective.full_eval(state.suffix_ids)
            metrics_full["step"] = state.step
            state.evals.append(metrics_full)
            eval_loss = metrics_full["loss"]
            improved = eval_loss < state.best_loss - 1e-9
            if improved:
                state.best_loss = eval_loss
                state.best_suffix_ids = list(state.suffix_ids)
                state.evals_without_improvement = 0
            else:
                state.evals_without_improvement += 1
            log.info(
                "eval @ step %d | full-pool L=%.5f MI=%.5f help=%.3f util_kl=%.5f "
                "validity=%.4f | best L=%.5f%s",
                state.step,
                eval_loss,
                metrics_full["mi"],
                metrics_full["help_rate"],
                metrics_full["util_kl"],
                metrics_full["validity_independent"],
                state.best_loss,
                " (new best)" if improved else "",
            )
            evals_log.write(
                {
                    "step": state.step,
                    "is_best": int(improved),
                    **{k: v for k, v in metrics_full.items() if k in EVAL_FIELDS},
                }
            )
            save_checkpoint(run_dir, cfg, state, rng, model, label)
            if state.evals_without_improvement >= patience:
                log.info(
                    "early stop: %d evals without improvement", state.evals_without_improvement
                )
                break

    steps_log.close()
    evals_log.close()
    log.info(
        "%s finished at step %d in %.1f min; best L=%.5f; best suffix=%r",
        label,
        state.step,
        timer.elapsed / 60,
        state.best_loss,
        model.decode_suffix(state.best_suffix_ids),
    )
    return state


# --- checkpoints -------------------------------------------------------------

def save_checkpoint(
    run_dir: str, cfg, state: GCGState, rng: np.random.Generator, model: MoveModel, label: str
) -> str:
    path = os.path.join(run_dir, f"{label}_checkpoint.pt")
    torch.save(
        {
            "step": state.step,
            "suffix_ids": state.suffix_ids,
            "best_suffix_ids": state.best_suffix_ids,
            "best_loss": state.best_loss,
            "evals_without_improvement": state.evals_without_improvement,
            "evals": state.evals,
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "config": cfg.as_dict(),
            "label": label,
        },
        path,
    )
    io_utils.write_text(
        os.path.join(run_dir, f"{label}_best_suffix.txt"),
        model.decode_suffix(state.best_suffix_ids) + "\n",
    )
    io_utils.write_json(
        os.path.join(run_dir, f"{label}_best_suffix.json"),
        {
            "suffix": model.decode_suffix(state.best_suffix_ids),
            "suffix_ids": state.best_suffix_ids,
            "best_loss": state.best_loss,
            "step": state.step,
            "model": cfg.model.name,
            "suffix_position": cfg.prompts.suffix_position,
            "seed": cfg.run.seed,
        },
    )
    return path


def load_checkpoint(
    path: str, rng: np.random.Generator
) -> Tuple[GCGState, np.random.Generator]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = GCGState(
        step=payload["step"],
        suffix_ids=list(payload["suffix_ids"]),
        best_suffix_ids=list(payload["best_suffix_ids"]),
        best_loss=float(payload["best_loss"]),
        evals_without_improvement=int(payload["evals_without_improvement"]),
        evals=list(payload["evals"]),
    )
    rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if payload.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng_state"]])
    return state, rng


# --- top-level runner --------------------------------------------------------

def run_gcg(
    cfg,
    resume_run_id: Optional[str] = None,
    skip_sanity: bool = False,
    label: str = "gcg",
) -> Dict[str, object]:
    """Sanity-check, then optimise the defensive suffix. Resumable by run_id."""
    if resume_run_id:
        cfg.run.run_id = resume_run_id
    run_dir = cfg.run_dir()
    io_utils.setup_logging(run_dir)
    cfg.save(run_dir)
    model = MoveModel(
        cfg.model.name,
        device=cfg.run.device,
        dtype=cfg.run.dtype,
        moves=cfg.game.moves,
        batch_size=cfg.model.batch_size,
        use_kv_cache=cfg.model.use_kv_cache,
        max_batch_size=cfg.model.max_batch_size,
    )
    log.info("%s | run_dir=%s", model.move_tokens.report(), run_dir)
    torch.manual_seed(cfg.run.seed)

    if not skip_sanity:
        report = run_sanity(cfg, model=model, run_dir=run_dir)
        if not report.passed:
            raise RuntimeError(
                "sanity checks failed; strengthen the prompts before optimising"
            )

    datasets = build_datasets(cfg)
    colluding = build_pool(
        model, "colluding", datasets.train["colluding"], cfg.prompts.suffix_position
    )
    independent = build_pool(
        model, "independent", datasets.train["independent"], cfg.prompts.suffix_position
    )
    direction, mu_independent = load_direction(cfg, model, independent)
    objective = MIDefenceObjective(
        model, cfg, colluding, independent, direction, mu_independent
    )

    rng = np.random.default_rng(cfg.run.seed)
    state = None
    checkpoint = os.path.join(run_dir, f"{label}_checkpoint.pt")
    if resume_run_id and os.path.exists(checkpoint):
        state, rng = load_checkpoint(checkpoint, rng)
        log.info("resumed %s from step %d (best L=%.5f)", resume_run_id, state.step, state.best_loss)
    elif resume_run_id:
        log.warning("no checkpoint at %s; starting from scratch", checkpoint)

    state = optimise(model, cfg, objective, run_dir, state=state, rng=rng, label=label)
    baseline = objective.full_eval([])  # empty suffix -> the slot-free prompts
    best = objective.full_eval(state.best_suffix_ids)
    with io_utils.metrics_log(run_dir, f"{label}_train_metrics.csv") as csv_log:
        for method, metrics in (("none", baseline), ("gcg_suffix", best)):
            for key, value in metrics.items():
                csv_log.write(
                    {
                        "method": method,
                        "method_params": f"lambda_util={cfg.loss.lambda_util}",
                        "seed": cfg.run.seed,
                        "model": cfg.model.name,
                        "split": "train",
                        "condition": "pooled",
                        "metric": key,
                        "value": value,
                    }
                )
    log.info(
        "train pool | no defence: MI=%.5f help=%.3f | suffix: MI=%.5f help=%.3f util_kl=%.5f",
        baseline["mi"],
        baseline["help_rate"],
        best["mi"],
        best["help_rate"],
        best["util_kl"],
    )
    return {
        "run_id": cfg.resolved_run_id(),
        "run_dir": run_dir,
        "suffix": model.decode_suffix(state.best_suffix_ids),
        "suffix_ids": state.best_suffix_ids,
        "best_loss": state.best_loss,
        "steps": state.step,
        "baseline_train": baseline,
        "suffix_train": best,
    }


def load_direction(cfg, model: MoveModel, independent: PromptPool):
    """Load r_hat (unit vector) and mu_indep from `collusion_direction_path`."""
    if cfg.loss.lambda_repr <= 0 or not cfg.loss.collusion_direction_path:
        return None, None
    from .direction import load_direction_file

    direction, layer, mu, _, _ = load_direction_file(cfg.loss.collusion_direction_path)
    if cfg.loss.repr_layer is None:
        cfg.loss.repr_layer = layer
    log.info(
        "collusion direction: layer=%d |r|=%.4f mu_indep=%.4f",
        cfg.loss.repr_layer,
        float(direction.norm()),
        mu,
    )
    return direction.to(model.device), mu
