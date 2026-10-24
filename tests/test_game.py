"""The payoff table and the collusion target f(a_A).

The expected payoffs are Table 1 of Bonjour, Aggarwal & Bhargava (UAI 2022),
transcribed by hand from the paper, so this test pins our implementation to
the published environment rather than to itself.
"""

import itertools

import pytest

from gcg_defense import game
from gcg_defense.game import PAPER, ROCK, SCISSORS

# BONJOUR_TABLE[p3][p1][p2] = (payoff P1, payoff P2, payoff P3)
BONJOUR_TABLE = {
    ROCK: {
        ROCK: {ROCK: (0, 0, 0), PAPER: (0, 1, 0), SCISSORS: (1, 0, 1)},
        PAPER: {ROCK: (1, 0, 0), PAPER: (1, 1, 0), SCISSORS: (1, 1, 1)},
        SCISSORS: {ROCK: (0, 1, 1), PAPER: (1, 1, 1), SCISSORS: (0, 0, 1)},
    },
    PAPER: {
        ROCK: {ROCK: (0, 0, 1), PAPER: (0, 1, 1), SCISSORS: (1, 1, 1)},
        PAPER: {ROCK: (1, 0, 1), PAPER: (0, 0, 0), SCISSORS: (0, 1, 0)},
        SCISSORS: {ROCK: (1, 1, 1), PAPER: (1, 0, 0), SCISSORS: (1, 1, 0)},
    },
    SCISSORS: {
        ROCK: {ROCK: (1, 1, 0), PAPER: (1, 1, 1), SCISSORS: (1, 0, 0)},
        PAPER: {ROCK: (1, 1, 1), PAPER: (0, 0, 1), SCISSORS: (0, 1, 1)},
        SCISSORS: {ROCK: (0, 1, 0), PAPER: (1, 0, 1), SCISSORS: (0, 0, 0)},
    },
}


def test_payoffs_match_bonjour_table_1():
    for p3, block in BONJOUR_TABLE.items():
        for p1, row in block.items():
            for p2, expected in row.items():
                assert game.payoffs(p1, p2, p3) == expected, (p1, p2, p3)


def test_payoff_rules_are_consistent():
    for a, b, c in itertools.product(range(3), repeat=3):
        pay = game.payoffs(a, b, c)
        distinct = len({a, b, c})
        if distinct == 1:
            assert pay == (0, 0, 0)
        elif distinct == 3:
            assert pay == (1, 1, 1)
        else:
            assert sum(pay) in (1, 2)


@pytest.mark.parametrize("move_a", [ROCK, PAPER, SCISSORS])
def test_collusion_target_guarantees_a_a_point(move_a):
    """f(a_A) gives A a point for EVERY move of C, and is the only such move."""
    target = game.collusion_target(move_a)
    for move_c in range(3):
        assert game.payoffs(move_a, target, move_c)[0] == 1
    guaranteed = [b for b in range(3) if game.guarantees_point_for_a(move_a, b)]
    assert guaranteed == [target]


def test_collusion_target_values():
    assert game.collusion_target(ROCK) == SCISSORS
    assert game.collusion_target(PAPER) == ROCK
    assert game.collusion_target(SCISSORS) == PAPER


def test_expected_payoffs_against_uniform_opponents():
    uniform = [1 / 3] * 3
    # RPS is symmetric under a cyclic relabelling, so every move of B has the
    # same expected payoff against uniform A and C.
    per_move = game.payoff_b_given_a().mean(axis=0)
    assert per_move.min() == pytest.approx(per_move.max())
    assert game.expected_payoff_b(uniform) == pytest.approx(5 / 9)
    assert game.best_response_gap(uniform) == pytest.approx(0.0, abs=1e-12)


def test_constant_policy_is_exploitable():
    rock = [1.0, 0.0, 0.0]
    assert game.best_response_gap(rock) > 0.1


def test_format_history_round_trip():
    text = game.format_history([game.round_record(0, 2, 1)])
    assert "Rock" in text and "Scissors" in text and "Paper" in text
    assert game.format_history([]).startswith("No rounds")
