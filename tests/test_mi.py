"""The closed-form mutual-information term on toy distributions."""

import math

import pytest
import torch

from gcg_defense import game, loss


def test_independent_gives_zero():
    q = torch.full((4, 3, 3), 1 / 3)
    assert loss.mutual_information(q).abs().max() < 1e-7


def test_identical_rows_give_zero_even_if_skewed():
    row = torch.tensor([0.7, 0.2, 0.1])
    q = row.expand(3, 3).unsqueeze(0).clone()
    assert loss.mutual_information(q).item() == pytest.approx(0.0, abs=1e-7)


def test_deterministic_copy_gives_log_three():
    q = torch.eye(3).unsqueeze(0)
    assert loss.mutual_information(q).item() == pytest.approx(math.log(3), abs=1e-5)


def test_deterministic_collusion_gives_log_three():
    """q_a = delta_{f(a)} is a permutation of the moves, so I = log 3."""
    q = torch.zeros(1, 3, 3)
    for a in range(3):
        q[0, a, game.collusion_target(a)] = 1.0
    assert loss.mutual_information(q).item() == pytest.approx(math.log(3), abs=1e-5)


@pytest.mark.parametrize("cp", [0.01, 0.05, 0.1])
def test_partial_collusion_matches_cp_squared(cp):
    """B plays f(a) w.p. cp, uniform otherwise -> I(A;B) = cp^2 + O(cp^3) nats.

    This identity is what makes the residual-collusion certificate
    CP_resid ~ sqrt(lambda*/(2N)) a bound on an interpretable quantity.
    """
    q = torch.full((1, 3, 3), (1 - cp) / 3)
    for a in range(3):
        q[0, a, game.collusion_target(a)] += cp
    mi = loss.mutual_information(q).item()
    assert mi == pytest.approx(cp ** 2, rel=0.05)


def test_mi_is_monotone_in_collusion_strength():
    values = []
    for cp in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        q = torch.full((1, 3, 3), (1 - cp) / 3)
        for a in range(3):
            q[0, a, game.collusion_target(a)] += cp
        values.append(loss.mutual_information(q).item())
    assert all(b > a for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(math.log(3), abs=1e-5)


def test_mi_equals_conditional_mutual_information_definition():
    """Cross-check against I(A;B) = sum_ab p(a,b) log p(a,b)/(p(a)p(b))."""
    torch.manual_seed(0)
    q = torch.softmax(torch.randn(3, 3), dim=-1)
    joint = q / 3.0
    p_a = joint.sum(1, keepdim=True)
    p_b = joint.sum(0, keepdim=True)
    direct = (joint * (joint / (p_a * p_b)).log()).sum()
    assert loss.mutual_information(q.unsqueeze(0)).item() == pytest.approx(
        direct.item(), abs=1e-6
    )


def test_utility_loss_is_zero_without_a_change():
    q = torch.softmax(torch.randn(5, 3), dim=-1)
    util = loss.utility_loss(q, torch.ones(5), q)
    assert util.abs().max() < 1e-6


def test_utility_loss_penalises_invalid_answers():
    q = torch.softmax(torch.randn(5, 3), dim=-1)
    low = loss.utility_loss(q, torch.full((5,), 0.5), q)
    assert low.mean().item() == pytest.approx(math.log(2.0), abs=1e-5)


def test_help_rate_on_perfect_collusion():
    targets = torch.tensor([game.collusion_target(a) for a in range(3)])
    q = torch.zeros(1, 3, 3)
    for a in range(3):
        q[0, a, game.collusion_target(a)] = 1.0
    assert loss.help_rate(q, targets).item() == pytest.approx(1.0)
    uniform = torch.full((1, 3, 3), 1 / 3)
    assert loss.help_rate(uniform, targets).item() == pytest.approx(1 / 3)


def test_representation_loss_is_one_sided():
    proj = torch.tensor([[1.0, 2.0, 3.0]])
    assert loss.representation_loss(proj, 5.0).item() == pytest.approx(0.0)
    assert loss.representation_loss(proj, 0.0).item() == pytest.approx(
        (1 + 4 + 9) / 3
    )


def test_float32_resolution_on_small_mi():
    """MI values are ~1e-3 nats, so bfloat16 is too coarse to carry them.

    Rounding q once to bfloat16 already shifts MI by several percent at
    CP = 0.02; inside the model the error would compound over the logits and
    the softmax, which is why every probability is cast to float32 first.
    """
    cp = 0.02
    q = torch.full((1, 3, 3), (1 - cp) / 3)
    for a in range(3):
        q[0, a, game.collusion_target(a)] += cp
    exact = loss.mutual_information(q).item()
    coarse = loss.mutual_information(q.bfloat16()).item()
    assert exact == pytest.approx(cp ** 2, rel=0.05)
    assert abs(coarse - exact) > 0.01 * exact
