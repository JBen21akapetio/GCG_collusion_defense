"""The two detectors, the permutation calibration and the certificate."""

import math

import numpy as np
import pytest
from scipy import stats

from gcg_defense import detect, game


def _independent_record(n=400, seed=0, n_states=28):
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 3, size=(n, 3))
    states = rng.integers(0, n_states, size=n)
    return actions, states


def _colluding_record(n=400, cp=1.0, seed=0, n_states=28):
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 3, size=(n, 3))
    for t in range(n):
        if rng.random() < cp:
            actions[t, 1] = game.collusion_target(int(actions[t, 0]))
    states = rng.integers(0, n_states, size=n)
    return actions, states


def test_gamma_is_near_zero_for_independent_play_in_one_state():
    actions, _ = _independent_record(n=20000)
    states = np.zeros(len(actions), dtype=int)
    gammas = detect.gamma_matrix(actions, states, 1)
    assert gammas[0, 1] < 0.01
    assert gammas[0, 1] == pytest.approx(gammas[1, 0])


def test_gamma_is_log_three_for_perfect_collusion():
    actions, _ = _colluding_record(n=5000, cp=1.0)
    states = np.zeros(len(actions), dtype=int)
    gammas = detect.gamma_matrix(actions, states, 1)
    assert gammas[0, 1] == pytest.approx(math.log(3), abs=0.02)
    assert gammas[0, 2] < 0.01
    assert gammas[1, 2] < 0.01


def test_gamma_recovers_cp_squared_for_partial_collusion():
    actions, _ = _colluding_record(n=200000, cp=0.2)
    states = np.zeros(len(actions), dtype=int)
    gamma = detect.gamma_matrix(actions, states, 1)[0, 1]
    assert gamma == pytest.approx(0.2 ** 2, rel=0.1)


def test_net_influence_isolates_the_colluding_pair():
    actions, states = _colluding_record(n=4000, cp=1.0)
    gammas = detect.gamma_matrix(actions, states, 28)
    net = detect.net_influence(gammas)
    assert net[(0, 1)] > 0.5 and net[(1, 0)] > 0.5
    assert net[(0, 2)] < 0.1 and net[(1, 2)] < 0.1


def test_plugin_gamma_is_biased_upwards_on_short_records():
    """Why a fixed threshold is not enough: the estimator is inflated by df/2N."""
    actions, states = _independent_record(n=50, n_states=28)
    gamma = detect.gamma_matrix(actions, states, 28)[0, 1]
    assert gamma > 0.05  # above Bonjour's alpha even though play is independent


def test_permutation_test_controls_the_false_positive_rate():
    """The calibrated test must not fire on independent play, where a fixed
    threshold does (previous test)."""
    fired = 0
    trials = 40
    for seed in range(trials):
        actions, states = _independent_record(n=50, seed=seed, n_states=28)
        result = detect.detect_collusion(
            actions, states, 28, n_perm=200, rng=np.random.default_rng(seed)
        )
        fired += int(result.detected_ab)
    assert fired / trials <= 0.1


def test_permutation_test_detects_strong_collusion():
    actions, states = _colluding_record(n=200, cp=1.0)
    result = detect.detect_collusion(
        actions, states, 28, n_perm=200, rng=np.random.default_rng(0)
    )
    assert result.detected_ab
    assert result.p_values[(0, 1)] < 0.05 / 3
    assert not result.detected_pairs == [(0, 1), (0, 2), (1, 2)]


def test_permutation_null_preserves_marginals():
    actions, states = _colluding_record(n=300, cp=1.0)
    null = detect.permutation_gammas(
        actions[:, 0], actions[:, 1], states, 28, 50, np.random.default_rng(0)
    )
    observed = detect.gamma_matrix(actions, states, 28)[0, 1]
    assert null.shape == (50,)
    assert null.max() < observed
    assert (null >= 0).all()


def test_noncentrality_hits_the_requested_power():
    for df in (4, 28, 112):
        lam = detect.noncentrality_for_power(df, alpha=0.05 / 3, power=0.8)
        critical = stats.chi2.ppf(1 - 0.05 / 3, df)
        assert stats.ncx2.sf(critical, df, lam) == pytest.approx(0.8, abs=1e-3)


def test_certificate_shrinks_like_one_over_sqrt_n():
    small = detect.residual_certificate(100, 28)
    large = detect.residual_certificate(400, 28)
    assert large.cp_resid == pytest.approx(small.cp_resid / 2, rel=0.01)
    assert large.mi_bound < small.mi_bound
    assert small.df == 28 * 4


def test_certificate_is_a_bound_on_an_undetectable_cp():
    """A CP at the certificate is at the edge of detectability; 3x it is not."""
    cert = detect.residual_certificate(500, 28, n_pairs=3)
    detected = 0
    for seed in range(10):
        actions, states = _colluding_record(
            n=500, cp=min(1.0, 3 * cert.cp_resid), seed=seed
        )
        result = detect.detect_collusion(
            actions, states, 28, n_perm=200, rng=np.random.default_rng(seed)
        )
        detected += int(result.detected_ab)
    assert detected >= 7


def test_state_index_is_injective_and_covers_the_range():
    seen = {detect.state_index(None)}
    for a in range(3):
        for b in range(3):
            for c in range(3):
                seen.add(detect.state_index((a, b, c)))
    assert len(seen) == detect.N_STATES_LAST_JOINT
    assert max(seen) == detect.N_STATES_LAST_JOINT - 1
