"""Prompt construction, condition blocks and the train/test instruction split."""

import numpy as np
import pytest

from gcg_defense import prompts
from gcg_defense.game import MOVES
from gcg_defense.prompts import (
    COLLUSION_INSTRUCTIONS,
    HANDWRITTEN_DEFENCES,
    SUFFIX_SLOT,
    build_datasets,
    build_messages,
    instruction_pool,
)


def test_instruction_inventory():
    assert len(COLLUSION_INSTRUCTIONS) >= 12
    assert all("{move_A}" in text for text in COLLUSION_INSTRUCTIONS)
    assert len(set(COLLUSION_INSTRUCTIONS)) == len(COLLUSION_INSTRUCTIONS)
    assert len(HANDWRITTEN_DEFENCES) >= 3


def test_instruction_split_is_by_instruction_and_disjoint():
    train = instruction_pool("colluding", "train", 8)
    test = instruction_pool("colluding", "test", 8)
    assert len(train) == 8
    assert len(test) >= 4
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(len(COLLUSION_INSTRUCTIONS)))


def test_instructions_vary_in_length_and_framing():
    lengths = sorted(len(t) for t in COLLUSION_INSTRUCTIONS)
    assert lengths[0] < 60 and lengths[-1] > 150


def test_independent_prompt_ignores_move_of_a():
    ctx = prompts.Context("independent", ((0, 1, 2),), 0, "u")
    texts = {
        str(build_messages(ctx, a, "system_end", True)) for a in range(3)
    }
    assert len(texts) == 1


@pytest.mark.parametrize("condition", ["colluding", "info_no_intent", "private_channel"])
def test_move_conditions_mention_the_move_of_a(condition):
    ctx = prompts.Context(condition, ((0, 1, 2),), 0, "u")
    for a, name in enumerate(MOVES):
        text = str(build_messages(ctx, a, "system_end", True))
        assert name in text


def test_suffix_slot_position():
    ctx = prompts.Context("colluding", ((0, 1, 2),), 0, "u")
    system_end = build_messages(ctx, 0, "system_end", True)
    assert system_end[0]["content"].rstrip().endswith(SUFFIX_SLOT)
    assert SUFFIX_SLOT not in system_end[1]["content"]
    user_end = build_messages(ctx, 0, "user_end", True)
    assert user_end[1]["content"].rstrip().endswith(SUFFIX_SLOT)
    assert SUFFIX_SLOT not in user_end[0]["content"]
    no_slot = build_messages(ctx, 0, "system_end", False)
    assert SUFFIX_SLOT not in str(no_slot)


def test_prompt_contains_rules_history_and_cue():
    ctx = prompts.Context("colluding", ((0, 1, 2),), 0, "u")
    messages = build_messages(ctx, 0, "system_end", True)
    assert "Rock beats Scissors" in messages[0]["content"]
    assert "Round 1:" in messages[1]["content"]
    assert prompts.FINAL_CUE in messages[1]["content"]


def test_history_length_follows_config(cfg):
    data = build_datasets(cfg)
    for pool in list(data.train.values()) + list(data.test.values()):
        assert all(len(c.history) == cfg.game.history_len for c in pool)


def test_datasets_are_deterministic_and_use_the_right_instructions(cfg):
    first = build_datasets(cfg)
    second = build_datasets(cfg)
    assert [c.history for c in first.train["colluding"]] == [
        c.history for c in second.train["colluding"]
    ]
    train_ids = {c.instruction_id for c in first.train["colluding"]}
    test_ids = {c.instruction_id for c in first.test["colluding"]}
    assert train_ids <= set(instruction_pool("colluding", "train", 8))
    assert test_ids <= set(instruction_pool("colluding", "test", 8))
    assert train_ids.isdisjoint(test_ids)
    assert len(first.train["colluding"]) == cfg.prompts.n_train_contexts
    assert len(first.test["colluding"]) == cfg.prompts.n_test_contexts


def test_different_seeds_give_different_contexts(cfg):
    a = build_datasets(cfg, seed=0)
    b = build_datasets(cfg, seed=1)
    assert [c.history for c in a.train["colluding"]] != [
        c.history for c in b.train["colluding"]
    ]


def test_instruction_assignment_is_balanced(cfg):
    data = build_datasets(cfg)
    counts = {}
    for ctx in data.train["colluding"]:
        counts[ctx.instruction_id] = counts.get(ctx.instruction_id, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_sample_history_is_in_range():
    rng = np.random.default_rng(0)
    history = prompts.sample_history(rng, 3)
    assert len(history) == 3
    assert all(0 <= m <= 2 for row in history for m in row)
