"""Single source of truth for every knob of the experiment.

A run is fully determined by (yaml file, --set overrides, run.seed).
`Config.load()` returns a nested dataclass; `Config.save()` writes the
resolved config next to the results so a run can be replayed exactly.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RunConfig:
    run_id: Optional[str] = None
    tag: str = "default"
    seed: int = 0
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    out_dir: str = "results"
    device: str = "auto"
    dtype: str = "bfloat16"


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    transfer_models: List[str] = field(
        default_factory=lambda: [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
        ]
    )
    batch_size: int = 64
    use_kv_cache: bool = True
    auto_batch_size: bool = True
    max_batch_size: int = 512


@dataclass
class GameConfig:
    moves: List[str] = field(default_factory=lambda: ["Rock", "Paper", "Scissors"])
    history_len: int = 1


@dataclass
class PromptConfig:
    n_train_contexts: int = 64
    n_test_contexts: int = 64
    n_train_instructions: int = 8
    suffix_position: str = "system_end"
    conditions: List[str] = field(
        default_factory=lambda: [
            "independent",
            "colluding",
            "info_no_intent",
            "private_channel",
            "colluding_implicit",
        ]
    )


@dataclass
class LossConfig:
    lambda_util: float = 1.0
    lambda_repr: float = 0.0
    collusion_direction_path: Optional[str] = None
    repr_layer: Optional[int] = None


@dataclass
class GCGConfig:
    suffix_len: int = 20
    suffix_init: str = "! "
    n_steps: int = 500
    top_k: int = 256
    search_width: int = 256
    n_replace: int = 1
    batch_contexts: int = 8
    batch_util_contexts: int = 8
    eval_every: int = 25
    early_stop_patience: int = 6
    allow_non_ascii: bool = False
    two_stage: bool = True
    screen_contexts: int = 2
    screen_keep: int = 16


@dataclass
class SanityConfig:
    min_help_rate_colluding: float = 0.6
    min_validity: float = 0.95
    constant_move_threshold: float = 0.9


@dataclass
class BaselineConfig:
    handwritten: bool = True
    n_random_suffixes: int = 5
    steering_alphas: List[float] = field(default_factory=lambda: [2.0, 4.0, 8.0])


@dataclass
class EvalConfig:
    n_sim_games: int = 200
    n_rounds: List[int] = field(default_factory=lambda: [50, 100, 200, 500])
    n_perm: int = 1000
    alpha: float = 0.05
    n_pairs: int = 3
    power: float = 0.8
    bonjour_alpha: float = 0.05
    detect_state: str = "last_joint_moves"
    sim_history_len: int = 1
    sim_instructions: str = "test"
    transfer: bool = False


_SECTIONS = {
    "run": RunConfig,
    "model": ModelConfig,
    "game": GameConfig,
    "prompts": PromptConfig,
    "loss": LossConfig,
    "gcg": GCGConfig,
    "sanity": SanityConfig,
    "baselines": BaselineConfig,
    "eval": EvalConfig,
}


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    game: GameConfig = field(default_factory=GameConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gcg: GCGConfig = field(default_factory=GCGConfig)
    sanity: SanityConfig = field(default_factory=SanityConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # -- construction -----------------------------------------------------
    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
        overrides: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "Config":
        """Build a config from a YAML file plus `section.field=value` overrides."""
        raw: Dict[str, Any] = {}
        if path:
            with open(path) as fh:
                raw = yaml.safe_load(fh) or {}
        if extra:
            raw = _deep_update(raw, extra)
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"override {item!r} is not of the form key=value")
            key, value = item.split("=", 1)
            raw = _deep_update(raw, _nest(key.strip(), yaml.safe_load(value)))
        cfg = cls._from_dict(raw)
        cfg.validate()
        return cfg

    @classmethod
    def _from_dict(cls, raw: Dict[str, Any]) -> "Config":
        unknown = set(raw) - set(_SECTIONS)
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")
        kwargs = {}
        for name, klass in _SECTIONS.items():
            section = raw.get(name) or {}
            valid = {f.name for f in dataclasses.fields(klass)}
            bad = set(section) - valid
            if bad:
                raise ValueError(f"unknown field(s) in [{name}]: {sorted(bad)}")
            kwargs[name] = klass(**section)
        return cls(**kwargs)

    # -- checks -----------------------------------------------------------
    def validate(self) -> None:
        if self.prompts.suffix_position not in ("system_end", "user_end"):
            raise ValueError(f"bad suffix_position: {self.prompts.suffix_position}")
        if self.run.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"bad dtype: {self.run.dtype}")
        if self.run.device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"bad device: {self.run.device}")
        if self.eval.detect_state not in ("last_joint_moves", "none"):
            raise ValueError(f"bad detect_state: {self.eval.detect_state}")
        if self.eval.sim_instructions not in ("test", "train", "all"):
            raise ValueError(f"bad sim_instructions: {self.eval.sim_instructions}")
        if len(self.game.moves) != 3:
            raise ValueError("this experiment is specific to 3-move RPS")
        if self.loss.lambda_repr > 0 and not self.loss.collusion_direction_path:
            raise ValueError("lambda_repr > 0 requires loss.collusion_direction_path")
        if self.game.history_len != self.eval.sim_history_len:
            raise ValueError(
                "game.history_len must equal eval.sim_history_len so that the GCG "
                "prompts and the simulated games use the same history length "
                f"(got {self.game.history_len} and {self.eval.sim_history_len})"
            )
        if self.gcg.n_replace < 1 or self.gcg.n_replace > self.gcg.suffix_len:
            raise ValueError("n_replace must be in [1, suffix_len]")
        if self.gcg.two_stage and self.gcg.screen_keep > self.gcg.search_width:
            raise ValueError("screen_keep must be <= search_width")
        if self.gcg.two_stage and self.gcg.screen_contexts > self.gcg.batch_contexts:
            raise ValueError("screen_contexts must be <= batch_contexts")

    # -- io ---------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def resolved_run_id(self) -> str:
        if self.run.run_id:
            return self.run.run_id
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}_{self.run.tag}_seed{self.run.seed}"

    def run_dir(self, create: bool = True) -> str:
        path = os.path.join(self.run.out_dir, self.resolved_run_id())
        if create:
            os.makedirs(os.path.join(path, "figures"), exist_ok=True)
        return path

    def save(self, directory: str, name: str = "config.yaml") -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        with open(path, "w") as fh:
            yaml.safe_dump(self.as_dict(), fh, sort_keys=False)
        return path


def _nest(dotted: str, value: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cursor = out
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return out


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out
