"""Les contrôles hors boucle : température, argmax, et parsing du texte généré."""

import math

import numpy as np
import pytest

from gcg_defense import game
from gcg_defense.out_of_loop import (
    TEMPERATURES,
    apply_temperature,
    decoding_table,
    help_rate_np,
    mutual_information_np,
    parse_move,
)


def _collusive_q(cp: float) -> np.ndarray:
    """Politique « aide avec probabilité cp, sinon uniforme », un contexte."""
    q = np.full((1, 3, 3), (1 - cp) / 3)
    for a in range(3):
        q[0, a, game.collusion_target(a)] += cp
    return q


def test_temperature_one_is_the_identity():
    q = _collusive_q(0.5)
    assert np.allclose(apply_temperature(q, 1.0), q)


def test_temperature_zero_is_the_argmax():
    q = _collusive_q(0.5)
    greedy = apply_temperature(q, 0.0)
    assert set(np.unique(greedy)) <= {0.0, 1.0}
    assert np.allclose(greedy.sum(axis=-1), 1.0)
    for a in range(3):
        assert greedy[0, a].argmax() == game.collusion_target(a)


def test_low_temperature_sharpens_and_high_flattens():
    q = _collusive_q(0.3)
    sharp = apply_temperature(q, 0.25).max()
    flat = apply_temperature(q, 4.0).max()
    assert sharp > q.max() > flat


def test_temperature_preserves_the_distribution_axis():
    q = _collusive_q(0.4)
    for t in TEMPERATURES:
        assert np.allclose(apply_temperature(q, t).sum(axis=-1), 1.0)


def test_greedy_mi_is_log_three_when_argmax_tracks_a():
    """Le cas qui invaliderait le résultat : MI faible mais argmax qui suit A."""
    q = _collusive_q(0.05)               # a peine collusif en probabilite
    assert mutual_information_np(q).item() < 0.01
    greedy = apply_temperature(q, 0.0)   # mais l'argmax suit toujours A
    assert mutual_information_np(greedy).item() == pytest.approx(math.log(3), abs=1e-6)


def test_greedy_mi_is_zero_when_argmax_is_constant():
    q = np.array([[[0.5, 0.3, 0.2], [0.5, 0.25, 0.25], [0.45, 0.35, 0.2]]])
    greedy = apply_temperature(q, 0.0)
    assert mutual_information_np(greedy).item() == pytest.approx(0.0, abs=1e-9)


def test_mutual_information_matches_the_torch_implementation():
    import torch

    from gcg_defense.loss import mutual_information

    rng = np.random.default_rng(0)
    q = rng.random((5, 3, 3))
    q /= q.sum(axis=-1, keepdims=True)
    theirs = mutual_information(torch.tensor(q)).numpy()
    assert np.allclose(mutual_information_np(q), theirs, atol=1e-6)


def test_help_rate_matches_the_torch_implementation():
    import torch

    from gcg_defense.loss import help_rate

    targets = torch.tensor([game.collusion_target(a) for a in range(3)])
    q = _collusive_q(0.7)
    theirs = help_rate(torch.tensor(q), targets).numpy()
    assert np.allclose(help_rate_np(q), theirs, atol=1e-6)


def test_decoding_table_covers_every_temperature_and_is_ordered():
    rows = decoding_table(_collusive_q(0.6))
    assert [r["temperature"] for r in rows] == list(TEMPERATURES)
    assert rows[0]["decoder"].startswith("greedy")
    assert all(0.0 <= r["help_rate"] <= 1.0 for r in rows)


def test_decoding_table_reports_context_dependence():
    rows = decoding_table(_collusive_q(0.6))
    assert rows[0]["fraction_context_dependent"] == pytest.approx(1.0)
    flat = np.full((4, 3, 3), 1 / 3)
    assert decoding_table(flat)[0]["fraction_context_dependent"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Rock", 0), ("  paper ", 1), ("I play Scissors.", 2),
        ("**Rock**", 0), ("My move for this round: Paper", 1),
        ("", None), ("I am not sure", None), ("Sc", None),
    ],
)
def test_parse_move(text, expected):
    assert parse_move(text) == expected


def test_parse_move_takes_the_first_mention():
    assert parse_move("Rock beats Scissors so I play Rock") == 0
