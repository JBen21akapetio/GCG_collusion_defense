"""Cost and duration model for one episode, and for a sweep of episodes.

An **episode** is one GCG run: one config overlay x one seed, on one GPU, in
one container. Its wall-clock cost is

    duration = startup + model_load + n_steps * s_per_step
               + n_evals * eval_seconds (+ evaluation stage, if run in the same
                                          container)
    cost     = gpu_price_per_second * duration
               + cpu_cores * cpu_price + memory_gib * memory_price (same duration)

`s_per_step` must come from a **measured** benchmark (`scripts/estimate_cost.py
--step model --steps-csv ...`), never from a guess. Early stopping is ignored:
every estimate is an upper bound at the full `n_steps`.

Prices are read from `configs/pricing.json`, which records the URL and the date
they were fetched. Nothing here is hardcoded from memory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

DEFAULT_PRICING = os.path.join("configs", "pricing.json")

#: Conservative container overheads, in seconds. Startup covers image pull and
#: container boot; model_load covers weights coming off the HF cache volume.
#: Both are measured in the benchmark step and can be overridden.
DEFAULT_STARTUP_SECONDS = 30.0
DEFAULT_MODEL_LOAD_SECONDS = 45.0


@dataclass
class Pricing:
    gpu_per_second: Dict[str, float]
    cpu_per_core_second: float
    memory_per_gib_second: float
    gpu_memory_gib: Dict[str, int] = field(default_factory=dict)
    source: str = ""
    fetched: str = ""

    @classmethod
    def load(cls, path: str = DEFAULT_PRICING) -> "Pricing":
        with open(path) as fh:
            raw = json.load(fh)
        return cls(
            gpu_per_second=raw["gpu_per_second"],
            cpu_per_core_second=raw["cpu_per_core_second"],
            memory_per_gib_second=raw["memory_per_gib_second"],
            gpu_memory_gib=raw.get("gpu_memory_gib", {}),
            source=raw.get("_source", ""),
            fetched=raw.get("_fetched", ""),
        )

    def gpu_price(self, gpu: str) -> float:
        if gpu not in self.gpu_per_second:
            raise KeyError(
                f"no price for GPU {gpu!r}; known: {sorted(self.gpu_per_second)}"
            )
        return self.gpu_per_second[gpu]

    def provenance(self) -> str:
        return f"prices from {self.source} fetched {self.fetched}"


@dataclass
class Episode:
    """One planned GCG run."""

    name: str
    gpu: str
    seed: int
    config: str
    overrides: List[str] = field(default_factory=list)
    n_steps: int = 500
    eval_every: int = 25
    s_per_step: Optional[float] = None
    eval_seconds: float = 0.0
    startup_seconds: float = DEFAULT_STARTUP_SECONDS
    model_load_seconds: float = DEFAULT_MODEL_LOAD_SECONDS
    evaluation_seconds: float = 0.0
    cpu_cores: float = 1.0
    memory_gib: float = 8.0

    @property
    def n_evals(self) -> int:
        return max(1, self.n_steps // max(1, self.eval_every))

    def duration_seconds(self) -> float:
        if self.s_per_step is None:
            raise ValueError(
                f"episode {self.name!r} has no measured s_per_step; run the "
                "benchmark step first (scripts/estimate_cost.py --step benchmark)"
            )
        return (
            self.startup_seconds
            + self.model_load_seconds
            + self.n_steps * self.s_per_step
            + self.n_evals * self.eval_seconds
            + self.evaluation_seconds
        )

    def cost(self, pricing: Pricing) -> float:
        seconds = self.duration_seconds()
        gpu = pricing.gpu_price(self.gpu) * seconds
        cpu = pricing.cpu_per_core_second * self.cpu_cores * seconds
        memory = pricing.memory_per_gib_second * self.memory_gib * seconds
        return gpu + cpu + memory

    def timeout_seconds(self, safety: float = 1.5) -> int:
        """Function timeout: estimated duration x safety, rounded up to a minute."""
        import math

        return int(math.ceil(self.duration_seconds() * safety / 60.0) * 60)


def summarise(
    episodes: Sequence[Episode],
    pricing: Pricing,
    max_concurrent: Optional[int] = None,
) -> str:
    """The episode | GPU | s/step | duration | cost table, with totals."""
    header = (
        f"{'episode':<34}{'GPU':>10}{'s/step':>9}{'duration':>11}"
        f"{'timeout':>10}{'cost':>9}"
    )
    lines = [header, "-" * len(header)]
    total_cost = 0.0
    total_seconds = 0.0
    by_gpu: Dict[str, List[float]] = {}
    for episode in episodes:
        seconds = episode.duration_seconds()
        cost = episode.cost(pricing)
        total_cost += cost
        total_seconds += seconds
        by_gpu.setdefault(episode.gpu, []).append(cost)
        lines.append(
            f"{episode.name:<34}{episode.gpu:>10}{episode.s_per_step:>9.2f}"
            f"{_hms(seconds):>11}{_hms(episode.timeout_seconds()):>10}"
            f"{'$' + format(cost, '.2f'):>9}"
        )
    lines.append("-" * len(header))
    for gpu, costs in sorted(by_gpu.items()):
        lines.append(
            f"{'subtotal ' + gpu:<34}{gpu:>10}{'':>9}{'':>11}{len(costs):>10}"
            f"{'$' + format(sum(costs), '.2f'):>9}"
        )
    lines.append(
        f"{'TOTAL (' + str(len(episodes)) + ' episodes)':<34}{'':>10}{'':>9}"
        f"{_hms(total_seconds):>11}{'':>10}{'$' + format(total_cost, '.2f'):>9}"
    )
    if max_concurrent:
        waves = (len(episodes) + max_concurrent - 1) // max_concurrent
        longest = max(e.duration_seconds() for e in episodes)
        lines.append(
            f"\nwall clock at max_concurrent={max_concurrent}: "
            f"{waves} wave(s) x ~{_hms(longest)} = ~{_hms(waves * longest)} "
            "(cost is unchanged by parallelism; only wall clock is)"
        )
    lines.append(f"\n{pricing.provenance()}")
    lines.append(
        "estimates assume the full n_steps: early stopping makes them an upper bound"
    )
    return "\n".join(lines)


def speed_per_dollar(
    s_per_step_by_gpu: Dict[str, float], pricing: Pricing
) -> str:
    """Which GPU gives the most GCG steps per dollar (the choice that matters)."""
    header = f"{'GPU':>10}{'s/step':>9}{'$/hour':>9}{'steps/$':>10}{'relative':>10}"
    lines = [header, "-" * len(header)]
    scores: Dict[str, float] = {}
    for gpu, s_per_step in s_per_step_by_gpu.items():
        price = pricing.gpu_price(gpu)
        steps_per_dollar = 1.0 / (price * s_per_step)
        scores[gpu] = steps_per_dollar
    best = max(scores.values()) if scores else 1.0
    for gpu, s_per_step in sorted(
        s_per_step_by_gpu.items(), key=lambda kv: -scores[kv[0]]
    ):
        price = pricing.gpu_price(gpu)
        lines.append(
            f"{gpu:>10}{s_per_step:>9.2f}{price * 3600:>9.2f}"
            f"{scores[gpu]:>10.0f}{scores[gpu] / best:>10.2f}"
        )
    lines.append(f"\n{pricing.provenance()}")
    return "\n".join(lines)


def s_per_step_from_csv(path: str, skip_first: int = 1) -> float:
    """Median seconds/step from a `*_steps.csv` written by the GCG loop.

    The first step is skipped: it pays for CUDA kernel autotuning and the
    allowed-token mask, and is not representative.
    """
    import csv
    import statistics

    values: List[float] = []
    with open(path) as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i < skip_first:
                continue
            try:
                values.append(float(row["seconds"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not values:
        raise ValueError(f"no usable 'seconds' column in {path}")
    return statistics.median(values)


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
