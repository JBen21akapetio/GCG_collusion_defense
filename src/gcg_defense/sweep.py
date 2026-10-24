"""The sweep grid: one episode per (config overlay, seed).

Both the launcher (`scripts/modal_app.py::sweep`) and the cost estimate
(`scripts/estimate_cost.py plan`) build their job list here, so a dry run and
an estimate can never describe a different sweep from the one that runs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .cost import Episode

#: The axes of the brief's sweep table, as config keys.
AXES = {
    "lambda_util": "loss.lambda_util",
    "lambda_repr": "loss.lambda_repr",
    "suffix_len": "gcg.suffix_len",
    "suffix_position": "prompts.suffix_position",
}


@dataclass
class SweepPoint:
    """One episode: a config, a set of overrides, a seed and a GPU."""

    tag: str
    seed: int
    gpu: str
    config: str
    overrides: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.tag}_seed{self.seed}"

    def override_string(self) -> str:
        """The `;`-separated form the Modal entry points take."""
        return ";".join(self.overrides)


def build_points(
    config: str = "configs/default.yaml",
    seeds: Sequence[int] = (0,),
    gpu: str = "A10G",
    axis_values: Optional[Dict[str, Sequence]] = None,
    extra_overrides: Sequence[str] = (),
) -> List[SweepPoint]:
    """The cartesian product of the requested axes with the seeds."""
    axis_values = {k: list(v) for k, v in (axis_values or {}).items() if v}
    unknown = set(axis_values) - set(AXES)
    if unknown:
        raise ValueError(f"unknown sweep axis/axes: {sorted(unknown)}")
    names = list(axis_values)
    combos = list(itertools.product(*(axis_values[n] for n in names))) or [()]
    points: List[SweepPoint] = []
    for combo in combos:
        parts = [f"{n}{v}" for n, v in zip(names, combo)]
        tag = "_".join(parts).replace(".", "p") if parts else "default"
        for seed in seeds:
            overrides = [f"run.seed={seed}", f"run.tag={tag}"]
            overrides += [f"{AXES[n]}={v}" for n, v in zip(names, combo)]
            overrides += list(extra_overrides)
            points.append(
                SweepPoint(
                    tag=tag, seed=int(seed), gpu=gpu, config=config, overrides=overrides
                )
            )
    return points


def to_episodes(
    points: Sequence[SweepPoint],
    s_per_step: Dict[str, float],
    n_steps: int = 500,
    eval_every: int = 25,
    eval_seconds: float = 0.0,
    evaluation_seconds: float = 0.0,
    startup_seconds: Optional[float] = None,
    model_load_seconds: Optional[float] = None,
) -> List[Episode]:
    """Attach measured timings to each point so it can be costed."""
    from .cost import DEFAULT_MODEL_LOAD_SECONDS, DEFAULT_STARTUP_SECONDS

    episodes: List[Episode] = []
    for point in points:
        if point.gpu not in s_per_step:
            raise ValueError(
                f"no measured s/step for {point.gpu!r}; run the benchmark step "
                f"(have: {sorted(s_per_step)})"
            )
        episodes.append(
            Episode(
                name=point.name,
                gpu=point.gpu,
                seed=point.seed,
                config=point.config,
                overrides=list(point.overrides),
                n_steps=n_steps,
                eval_every=eval_every,
                s_per_step=s_per_step[point.gpu],
                eval_seconds=eval_seconds,
                evaluation_seconds=evaluation_seconds,
                startup_seconds=(
                    DEFAULT_STARTUP_SECONDS if startup_seconds is None else startup_seconds
                ),
                model_load_seconds=(
                    DEFAULT_MODEL_LOAD_SECONDS
                    if model_load_seconds is None
                    else model_load_seconds
                ),
            )
        )
    return episodes


def format_plan(points: Sequence[SweepPoint], timeouts: Optional[Dict[str, int]] = None) -> str:
    """Human-readable list of the jobs a launch would create."""
    lines = [f"{len(points)} episode(s):", ""]
    for i, point in enumerate(points, start=1):
        timeout = (timeouts or {}).get(point.name)
        suffix = f"  timeout={timeout}s" if timeout else ""
        lines.append(f"  [{i:>2}] {point.name:<30} gpu={point.gpu:<9}{suffix}")
        lines.append(f"       config={point.config}")
        lines.append(f"       overrides={point.override_string()}")
    return "\n".join(lines)
