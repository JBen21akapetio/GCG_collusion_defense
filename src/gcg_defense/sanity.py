"""Sanity checks that must pass BEFORE any optimisation.

1. In `colluding`, the mean probability that B plays f(a_A) must be clearly
   above 1/3 (target: >= sanity.min_help_rate_colluding).
2. In `independent`, q must not depend on A's move (checked at the prompt
   level, exactly) and validity must be >= sanity.min_validity.
3. Baseline model-MI is reported for every condition.

`run_sanity` returns a report; callers stop the run when `passed` is False.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import io_utils
from .measure import measure_pool, scalar_metrics
from .model_utils import MoveModel
from .pools import build_pools
from .prompts import build_datasets, render

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class SanityReport:
    passed: bool
    checks: List[Check] = field(default_factory=list)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    move_tokens: str = ""

    def summary(self) -> str:
        lines = [self.move_tokens, ""]
        header = f"{'pool':<26}{'MI':>10}{'help':>8}{'valid':>8}{'H(q)':>8}{'payoffB':>9}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, m in self.metrics.items():
            lines.append(
                f"{name:<26}{m.get('model_mi', float('nan')):>10.5f}"
                f"{m.get('help_rate', float('nan')):>8.3f}"
                f"{m.get('validity', float('nan')):>8.3f}"
                f"{m.get('entropy', float('nan')):>8.3f}"
                f"{m.get('payoff_b', float('nan')):>9.3f}"
            )
        lines.append("")
        for check in self.checks:
            lines.append(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
        return "\n".join(lines)


def check_independent_prompt_invariance(model: MoveModel, datasets, cfg) -> Check:
    """The independent prompt must be byte-identical for all three moves of A."""
    bad = []
    for ctx in datasets.train["independent"][:8] + datasets.test["independent"][:8]:
        texts = {
            render(model.tokenizer, ctx, a, cfg.prompts.suffix_position, True).text()
            for a in range(3)
        }
        if len(texts) != 1:
            bad.append(ctx.uid)
    return Check(
        name="independent prompts ignore A's move",
        passed=not bad,
        detail="identical for all a" if not bad else f"differs for {bad}",
    )


def run_sanity(
    cfg,
    model: Optional[MoveModel] = None,
    run_dir: Optional[str] = None,
) -> SanityReport:
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
    log.info(model.move_tokens.report())
    datasets = build_datasets(cfg)
    pools = build_pools(model, datasets, cfg)

    report = SanityReport(passed=True, move_tokens=model.move_tokens.report())
    rows: List[Dict[str, object]] = []
    for split, split_pools in pools.items():
        for name, pool in split_pools.items():
            metrics = scalar_metrics(
                measure_pool(model, pool, use_base_prompts=True)
            )
            report.metrics[f"{split}/{name}"] = metrics
            for metric, value in metrics.items():
                rows.append(
                    {
                        "method": "none",
                        "method_params": "sanity",
                        "seed": cfg.run.seed,
                        "model": cfg.model.name,
                        "split": split,
                        "condition": name,
                        "metric": metric,
                        "value": value,
                    }
                )

    colluding = report.metrics["train/colluding"]
    help_rate = colluding.get("help_rate", 0.0)
    report.checks.append(
        Check(
            name="colluding help rate well above 1/3",
            passed=help_rate >= cfg.sanity.min_help_rate_colluding,
            detail=(
                f"mean q(f(a_A)) = {help_rate:.3f} "
                f"(target >= {cfg.sanity.min_help_rate_colluding}, chance = 0.333)"
            ),
        )
    )
    independent = report.metrics["train/independent"]
    report.checks.append(
        Check(
            name="independent validity",
            passed=independent["validity"] >= cfg.sanity.min_validity,
            detail=(
                f"mass on move tokens = {independent['validity']:.4f} "
                f"(target >= {cfg.sanity.min_validity})"
            ),
        )
    )
    report.checks.append(check_independent_prompt_invariance(model, datasets, cfg))
    report.checks.append(
        Check(
            name="B is not already constant",
            passed=independent["max_prob"] <= cfg.sanity.constant_move_threshold,
            detail=f"mean max_b q_b = {independent['max_prob']:.3f}",
        )
    )
    report.passed = all(c.passed for c in report.checks)

    if run_dir:
        with io_utils.metrics_log(run_dir, "sanity_metrics.csv") as csv_log:
            csv_log.write_all(rows)
        io_utils.write_text(
            f"{run_dir}/sanity_report.txt", report.summary() + "\n"
        )
    log.info("sanity report\n%s", report.summary())
    return report


def per_instruction_report(
    model: MoveModel, pool, suffix_ids=(), use_base_prompts: bool = True
) -> List[Dict[str, object]]:
    """Help / counter / copy rates per collusion instruction.

    help    : B plays f(a_A), the move that hands A a certain point;
    counter : B plays the move that beats A's move (self-interested reply);
    copy    : B mirrors A's move.
    """
    import numpy as np

    from . import game, loss
    from .measure import TARGETS

    prompts_ = pool.base_prompts if use_base_prompts else pool.prompts
    out = model.q_batch(prompts_, () if use_base_prompts else suffix_ids)
    q = pool.reshape(out["q"].float())           # [n, 3, 3]
    mi = loss.mutual_information(q)
    rows: List[Dict[str, object]] = []
    by_instruction: Dict[int, List[int]] = {}
    for i, ctx in enumerate(pool.contexts):
        by_instruction.setdefault(ctx.instruction_id, []).append(i)
    for instruction_id, idx in sorted(by_instruction.items()):
        sub = q[idx]
        help_rate = float(loss.help_rate(sub, TARGETS).mean())
        counter = float(
            np.mean([sub[:, a, game.LOSES_TO[a]].mean().item() for a in range(3)])
        )
        copy = float(np.mean([sub[:, a, a].mean().item() for a in range(3)]))
        rows.append(
            {
                "instruction_id": instruction_id,
                "n": len(idx),
                "help": help_rate,
                "counter": counter,
                "copy": copy,
                "mi": float(mi[idx].mean()),
                "entropy": float(loss.entropy(sub).mean()),
                "q": sub.mean(dim=0).tolist(),
            }
        )
    return rows


def format_instruction_report(rows: List[Dict[str, object]], texts) -> str:
    header = f"{'id':>3} {'n':>4} {'help':>7} {'counter':>8} {'copy':>7} {'MI':>8}  instruction"
    lines = [header, "-" * len(header)]
    for row in rows:
        text = texts[int(row["instruction_id"])]
        snippet = text if len(text) <= 60 else text[:57] + "..."
        lines.append(
            f"{row['instruction_id']:>3} {row['n']:>4} {row['help']:>7.3f} "
            f"{row['counter']:>8.3f} {row['copy']:>7.3f} {row['mi']:>8.4f}  {snippet}"
        )
    return "\n".join(lines)
