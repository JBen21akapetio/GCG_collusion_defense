"""Config loading, overrides, validation and run-directory naming."""

import os

import pytest
import yaml

from gcg_defense.config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_default_yaml_matches_the_brief(cfg):
    assert cfg.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.gcg.suffix_len == 20
    assert cfg.gcg.n_steps == 500
    assert cfg.gcg.top_k == 256
    assert cfg.gcg.search_width == 256
    assert cfg.gcg.n_replace == 1
    assert cfg.gcg.batch_contexts == 8
    assert cfg.gcg.batch_util_contexts == 8
    assert cfg.gcg.eval_every == 25
    assert cfg.gcg.early_stop_patience == 6
    assert cfg.gcg.allow_non_ascii is False
    assert cfg.gcg.suffix_init == "! "
    assert cfg.loss.lambda_util == 1.0
    assert cfg.loss.lambda_repr == 0.0
    assert cfg.prompts.n_train_contexts == 64
    assert cfg.prompts.n_test_contexts == 64
    assert cfg.prompts.n_train_instructions == 8
    assert cfg.prompts.suffix_position == "system_end"
    assert cfg.run.seeds == [0, 1, 2]
    assert cfg.run.dtype == "bfloat16"
    assert cfg.eval.n_sim_games == 200
    assert cfg.eval.n_rounds == [50, 100, 200, 500]
    assert cfg.eval.n_perm == 1000
    assert cfg.eval.alpha == 0.05
    assert cfg.eval.power == 0.8


def test_fast_profile_loads():
    fast = Config.load(os.path.join(ROOT, "configs", "fast.yaml"))
    assert fast.gcg.n_steps < 500
    assert fast.game.history_len == fast.eval.sim_history_len


def test_overrides_are_typed_and_nested():
    cfg = Config.load(
        os.path.join(ROOT, "configs", "default.yaml"),
        ["loss.lambda_util=10", "gcg.two_stage=false", "run.seed=7"],
    )
    assert cfg.loss.lambda_util == 10
    assert cfg.gcg.two_stage is False
    assert cfg.run.seed == 7


def test_unknown_fields_are_rejected():
    with pytest.raises(ValueError):
        Config._from_dict({"gcg": {"nonexistent": 1}})
    with pytest.raises(ValueError):
        Config._from_dict({"nonexistent_section": {}})


def test_history_lengths_must_agree():
    with pytest.raises(ValueError, match="sim_history_len"):
        Config.load(
            os.path.join(ROOT, "configs", "default.yaml"), ["game.history_len=3"]
        )


def test_repr_term_requires_a_direction_file():
    with pytest.raises(ValueError, match="collusion_direction_path"):
        Config.load(
            os.path.join(ROOT, "configs", "default.yaml"), ["loss.lambda_repr=1.0"]
        )


def test_bad_enums_are_rejected():
    for override in (
        "prompts.suffix_position=middle",
        "run.dtype=float8",
        "eval.detect_state=whatever",
        "gcg.n_replace=0",
    ):
        with pytest.raises(ValueError):
            Config.load(os.path.join(ROOT, "configs", "default.yaml"), [override])


def test_run_id_and_round_trip(tmp_path, cfg):
    cfg.run.out_dir = str(tmp_path)
    run_id = cfg.resolved_run_id()
    assert run_id.endswith("_default_seed0")
    directory = cfg.run_dir()
    assert os.path.isdir(os.path.join(directory, "figures"))
    path = cfg.save(directory)
    with open(path) as fh:
        reloaded = Config._from_dict(yaml.safe_load(fh))
    assert reloaded.as_dict() == cfg.as_dict()
