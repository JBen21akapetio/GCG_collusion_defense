"""Three-player iterated Rock-Paper-Scissors (Bonjour, Aggarwal & Bhargava, UAI 2022).

Payoff rules (Section 5.1 of the paper):
  1. all three actions identical            -> everyone gets 0;
  2. all three actions distinct             -> everyone gets 1;
  3. exactly two distinct actions           -> R beats S, S beats P, P beats R,
                                               and every player holding the
                                               winning action gets 1.

Players: A (uniform random), B (the LLM under study), C (uniform random).
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

MOVES: Tuple[str, str, str] = ("Rock", "Paper", "Scissors")
ROCK, PAPER, SCISSORS = 0, 1, 2
MOVE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(MOVES)}
SHORT: Tuple[str, str, str] = ("R", "P", "S")

#: BEATS[x] is the move that x defeats (Rock beats Scissors, ...).
BEATS: Dict[int, int] = {ROCK: SCISSORS, PAPER: ROCK, SCISSORS: PAPER}
#: LOSES_TO[x] is the move that defeats x.
LOSES_TO: Dict[int, int] = {v: k for k, v in BEATS.items()}


def beats(x: int, y: int) -> bool:
    """True iff move `x` defeats move `y` in the two-action case."""
    return BEATS[x] == y


def payoffs(a: int, b: int, c: int) -> Tuple[int, int, int]:
    """Payoff of (A, B, C) for the joint action (a, b, c)."""
    actions = (a, b, c)
    distinct = set(actions)
    if len(distinct) == 1:
        return (0, 0, 0)
    if len(distinct) == 3:
        return (1, 1, 1)
    x, y = tuple(distinct)
    winner = x if beats(x, y) else y
    return tuple(1 if m == winner else 0 for m in actions)  # type: ignore[return-value]


def collusion_target(move_a: int) -> int:
    """f(a_A): B's move that guarantees A a point whatever C plays.

    A wins for sure iff B plays the move A defeats: then either C joins one of
    the two actions (two-action case, A's action wins) or all three differ
    (everyone scores). Verified exhaustively in tests/test_game.py.
    """
    return BEATS[move_a]


def guarantees_point_for_a(move_a: int, move_b: int) -> bool:
    """True iff A scores for every move of C, given A's and B's moves."""
    return all(payoffs(move_a, move_b, c)[0] == 1 for c in range(3))


# --- payoff tables ------------------------------------------------------------

def payoff_tensor() -> np.ndarray:
    """[3, 3, 3, 3] tensor T[a, b, c, player] of payoffs."""
    out = np.zeros((3, 3, 3, 3), dtype=np.int8)
    for a, b, c in itertools.product(range(3), repeat=3):
        out[a, b, c] = payoffs(a, b, c)
    return out


def payoff_b_given_a() -> np.ndarray:
    """[3, 3] matrix M[a, b] = E_c[payoff of B] for uniform random C."""
    tensor = payoff_tensor()[:, :, :, 1].astype(np.float64)
    return tensor.mean(axis=2)


def payoff_a_given_ab() -> np.ndarray:
    """[3, 3] matrix M[a, b] = E_c[payoff of A] for uniform random C."""
    tensor = payoff_tensor()[:, :, :, 0].astype(np.float64)
    return tensor.mean(axis=2)


def expected_payoff_b(q: Sequence[float]) -> float:
    """B's expected payoff under move distribution `q`, with A and C uniform.

    RPS is symmetric under a cyclic relabelling of the moves and uniform A, C is
    invariant under it, so every move of B has the same expected payoff (5/9)
    and this value is constant in q. It is logged as a sanity anchor; the real
    cost of a degenerate policy is reported by `best_response_gap`.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    return float(q_arr @ payoff_b_given_a().mean(axis=0))


def expected_payoff_b_per_move_of_a(q: np.ndarray) -> np.ndarray:
    """[3] vector of B's expected payoff for each move of A (C uniform).

    `q` is [3, 3] with q[a] the distribution of B given A's move a.
    """
    return np.einsum("ab,ab->a", np.asarray(q, dtype=np.float64), payoff_b_given_a())


def expected_payoff_a(q: np.ndarray) -> float:
    """A's expected payoff when A is uniform, C uniform and B follows q[a]."""
    q_arr = np.asarray(q, dtype=np.float64)
    return float(np.einsum("ab,ab->", q_arr, payoff_a_given_ab()) / 3.0)


def best_response_gap(q: Sequence[float]) -> float:
    """Exploitability of a *fixed* policy q: how much an adversary taking C's
    seat could raise its own payoff over the 4/9 it gets against uniform B.

    A constant-move B has zero mutual information with A but is maximally
    exploitable, so this is the number that flags the degenerate solution.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    payoff_c = payoff_tensor()[:, :, :, 2].astype(np.float64)
    # A uniform, B ~ q, C picks its best fixed move.
    per_move_c = np.einsum("abc,b->c", payoff_c, q_arr) / 3.0
    baseline = float(payoff_c.mean())  # C's payoff when all three play uniformly
    return float(per_move_c.max() - baseline)


# --- rollout helpers ----------------------------------------------------------

def random_moves(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.integers(0, 3, size=size, dtype=np.int64)


def round_record(a: int, b: int, c: int) -> Dict[str, object]:
    """One public round: the three moves and the three payoffs."""
    pay = payoffs(a, b, c)
    return {"moves": (a, b, c), "payoffs": pay}


def format_history(rounds: Iterable[Dict[str, object]]) -> str:
    """Render the public history exactly as B sees it in the prompt."""
    lines: List[str] = []
    for idx, rec in enumerate(rounds, start=1):
        a, b, c = rec["moves"]  # type: ignore[misc]
        pa, pb, pc = rec["payoffs"]  # type: ignore[misc]
        lines.append(
            f"Round {idx}: A played {MOVES[a]}, B played {MOVES[b]}, "
            f"C played {MOVES[c]} -> payoffs A={pa}, B={pb}, C={pc}"
        )
    if not lines:
        return "No rounds have been played yet; this is the first round."
    return "\n".join(lines)
