"""Les figures explicatives : décomposition des coups et objets du jeu."""

import numpy as np
import pytest

from gcg_defense import game
from gcg_defense.explain import CONDITIONS, decomposition


def test_decomposition_of_perfect_collusion():
    """B joue toujours f(a_A) : toute la masse en 'help'."""
    matrix = np.zeros((3, 3))
    for a in range(3):
        matrix[a, game.collusion_target(a)] = 1.0
    parts = decomposition(matrix)
    assert parts["help"] == pytest.approx(1.0)
    assert parts["counter"] == pytest.approx(0.0)
    assert parts["copy"] == pytest.approx(0.0)


def test_decomposition_of_perfect_countering():
    """B joue toujours le coup qui bat A : toute la masse en 'counter'."""
    matrix = np.zeros((3, 3))
    for a in range(3):
        matrix[a, game.LOSES_TO[a]] = 1.0
    parts = decomposition(matrix)
    assert parts["counter"] == pytest.approx(1.0)
    assert parts["help"] == pytest.approx(0.0)


def test_decomposition_of_copying():
    matrix = np.eye(3)
    parts = decomposition(matrix)
    assert parts["copy"] == pytest.approx(1.0)


def test_the_three_shares_are_exclusive_and_sum_to_one():
    """help, counter et copy partitionnent les 3 coups : ils somment à 1."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        matrix = rng.random((3, 3))
        matrix /= matrix.sum(axis=1, keepdims=True)
        parts = decomposition(matrix)
        assert sum(parts.values()) == pytest.approx(1.0)


def test_uniform_policy_splits_evenly():
    parts = decomposition(np.full((3, 3), 1 / 3))
    for value in parts.values():
        assert value == pytest.approx(1 / 3)


def test_help_counter_and_copy_are_three_distinct_moves():
    """Le socle de la décomposition : pour chaque coup de A, les trois cibles
    sont distinctes, donc elles couvrent exactement les trois colonnes."""
    for a in range(3):
        targets = {game.collusion_target(a), game.LOSES_TO[a], a}
        assert len(targets) == 3


def test_conditions_listed_for_the_figures_exist_in_the_datasets(cfg):
    from gcg_defense.prompts import build_datasets

    data = build_datasets(cfg)
    for key, _label in CONDITIONS:
        assert key in data.test, f"{key} absent des pools de test"
