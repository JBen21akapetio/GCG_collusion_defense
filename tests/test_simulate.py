"""The lockstep simulator and its memoised policy table."""

import numpy as np
import pytest

from gcg_defense import detect, game
from gcg_defense.simulate import simulate


class FakePolicy:
    """Stands in for the memoised PolicyTable: q depends only on A's move."""

    def __init__(self, mode="collude"):
        self.mode = mode
        self.calls = 0
        self.keys_seen = set()

    def get(self, keys):
        self.calls += 1
        self.keys_seen.update(keys)
        out = np.zeros((len(keys), 3))
        for i, (_instr, _history, move_a) in enumerate(keys):
            if self.mode == "collude":
                out[i, game.collusion_target(move_a)] = 1.0
            elif self.mode == "constant":
                out[i, 0] = 1.0
            else:
                out[i] = 1 / 3
        return out

    def mean_validity(self):
        return 1.0

    n_forward = 0


def _run(mode, n_games=8, n_rounds=20, seed=0):
    return simulate(
        model=None,
        policy=FakePolicy(mode),
        instruction_ids=[0, 1],
        n_games=n_games,
        n_rounds=n_rounds,
        rng=np.random.default_rng(seed),
    )


def test_perfect_collusion_helps_a_every_round():
    sim = _run("collude")
    assert sim.help_fraction() == pytest.approx(1.0)
    assert sim.win_rates()[0] == pytest.approx(1.0)      # A always scores
    assert sim.a_advantage_over_c() > 0.3


def test_uniform_play_helps_a_third_of_the_time():
    sim = _run("uniform", n_games=40, n_rounds=200)
    assert sim.help_fraction() == pytest.approx(1 / 3, abs=0.03)
    assert sim.a_advantage_over_c() == pytest.approx(0.0, abs=0.03)


def test_payoffs_match_the_game_rules():
    sim = _run("uniform")
    for g in range(sim.actions.shape[0]):
        for t in range(sim.actions.shape[1]):
            a, b, c = sim.actions[g, t]
            assert tuple(sim.payoffs[g, t]) == game.payoffs(int(a), int(b), int(c))


def test_states_are_the_previous_round_joint_moves():
    sim = _run("uniform")
    for g in range(sim.actions.shape[0]):
        assert sim.states[g, 0] == detect.state_index(None)
        for t in range(1, sim.actions.shape[1]):
            assert sim.states[g, t] == detect.state_index(tuple(sim.actions[g, t - 1]))


def test_policy_is_queried_with_the_history_it_will_see():
    policy = FakePolicy("uniform")
    simulate(None, policy, [0], 4, 6, np.random.default_rng(0))
    assert policy.calls == 6  # one batched lookup per round
    histories = {key[1] for key in policy.keys_seen}
    assert None in histories  # the first round has no history
    assert all(h is None or len(h) == 3 for h in histories)


def test_simulation_is_deterministic_given_the_seed():
    first = _run("uniform", seed=3)
    second = _run("uniform", seed=3)
    assert np.array_equal(first.actions, second.actions)
    assert not np.array_equal(first.actions, _run("uniform", seed=4).actions)


def test_prefix_truncates_consistently():
    sim = _run("uniform", n_rounds=50)
    short = sim.prefix(10)
    assert short.actions.shape[1] == 10
    assert np.array_equal(short.actions, sim.actions[:, :10])
    assert np.array_equal(short.states, sim.states[:, :10])


def test_constant_policy_is_undetectable_but_degenerate():
    """A constant B carries no information about A -- the degenerate solution."""
    sim = _run("constant", n_games=4, n_rounds=300)
    result = detect.detect_collusion(
        sim.actions[0], sim.states[0], 28, n_perm=200, rng=np.random.default_rng(0)
    )
    assert not result.detected_ab
    assert sim.help_fraction() == pytest.approx(1 / 3, abs=0.06)
    assert game.best_response_gap([1.0, 0.0, 0.0]) > 0.1
