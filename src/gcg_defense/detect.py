"""Collusion detection from public game records.

Two detectors run on the same simulated games:

* **Bonjour net influence** (UAI 2022). The individual influence of i on j is
  the conditional mutual information of their actions given the public state,

      gamma(i;j) = sum_s p(s) sum_{ai,aj} p(ai,aj|s) log[ p(ai,aj|s) /
                                                          (p(ai|s) p(aj|s)) ],

  estimated by Monte Carlo counts, and the net influence subtracts the largest
  individual influence any other agent has on the target,

      Gamma(i;j) = gamma(i;j) - max_{k != i,j} gamma(k;j).

  Collusion is declared when Gamma(i;j) and Gamma(j;i) both exceed alpha.
  With a fixed alpha this is biased: the plug-in gamma is inflated by about
  df/(2N), so short records look collusive.

* **Our calibrated test.** The same gamma, but its null distribution is
  obtained by permuting one player's actions WITHIN each public state, which
  destroys any dependence between the two action sequences while preserving
  both policies and the state distribution. p-values are Bonferroni-corrected
  over the three unordered pairs.

When nothing is detected, `residual_certificate` reports how much collusion
could have hidden: the non-centrality lambda* that the test would catch with
80% power, turned into a collusion probability through I(A;B) = CP^2 + O(CP^3)
(see tests/test_mi.py), giving CP_resid <~ sqrt(lambda*/(2N)).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

log = logging.getLogger(__name__)

PLAYERS = ("A", "B", "C")
PAIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
EPS = 1e-300


# --- estimators ---------------------------------------------------------------

def joint_counts(
    actions_i: np.ndarray, actions_j: np.ndarray, states: np.ndarray, n_states: int
) -> np.ndarray:
    """[n_states, 3, 3] counts of (a_i, a_j) per state."""
    flat = states * 9 + actions_i * 3 + actions_j
    counts = np.bincount(flat, minlength=n_states * 9).astype(np.float64)
    return counts.reshape(n_states, 3, 3)


def gamma_from_counts(counts: np.ndarray) -> float:
    """Plug-in conditional mutual information in nats from per-state counts."""
    totals = counts.sum(axis=(1, 2), keepdims=True)
    n = counts.sum()
    if n <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        joint = np.where(totals > 0, counts / np.maximum(totals, 1.0), 0.0)
        p_i = joint.sum(axis=2, keepdims=True)
        p_j = joint.sum(axis=1, keepdims=True)
        ratio = np.where(joint > 0, joint / np.maximum(p_i * p_j, EPS), 1.0)
        per_state = (joint * np.log(np.maximum(ratio, EPS))).sum(axis=(1, 2))
    weights = totals.reshape(-1) / n
    return float((weights * per_state).sum())


def gamma_matrix(actions: np.ndarray, states: np.ndarray, n_states: int) -> np.ndarray:
    """[3, 3] symmetric matrix of individual influences (diagonal = 0)."""
    out = np.zeros((3, 3))
    for i, j in PAIRS:
        value = gamma_from_counts(
            joint_counts(actions[:, i], actions[:, j], states, n_states)
        )
        out[i, j] = out[j, i] = value
    return out


def net_influence(gammas: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Gamma(i;j) for every ordered pair (3 players: one 'other' agent each)."""
    out: Dict[Tuple[int, int], float] = {}
    for i, j in itertools.permutations(range(3), 2):
        others = [k for k in range(3) if k not in (i, j)]
        out[(i, j)] = float(gammas[i, j] - max(gammas[k, j] for k in others))
    return out


def permutation_gammas(
    actions_i: np.ndarray,
    actions_j: np.ndarray,
    states: np.ndarray,
    n_states: int,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null distribution of gamma under within-state permutation of j's actions.

    Vectorised over permutations: positions are grouped by state once, then
    each stratum is shuffled for all `n_perm` permutations at a time with
    `Generator.permuted`, which is O(n_perm * T) rather than the O(T log T)
    sort a lexsort-based implementation would pay.
    """
    order = np.argsort(states, kind="stable")
    grouped_states = states[order]
    grouped_i = actions_i[order]
    grouped_j = actions_j[order]
    n = grouped_states.size
    if n == 0:
        return np.zeros(n_perm)
    permuted_j = np.empty((n_perm, n), dtype=np.int64)
    _, starts = np.unique(grouped_states, return_index=True)
    bounds = list(starts) + [n]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        block = grouped_j[start:stop]
        if stop - start == 1:
            permuted_j[:, start:stop] = block
        else:
            permuted_j[:, start:stop] = rng.permuted(
                np.broadcast_to(block, (n_perm, stop - start)), axis=1
            )
    flat = (
        np.arange(n_perm)[:, None] * (n_states * 9)
        + grouped_states[None, :] * 9
        + grouped_i[None, :] * 3
        + permuted_j
    )
    counts = np.bincount(flat.reshape(-1), minlength=n_perm * n_states * 9)
    counts = counts.astype(np.float64).reshape(n_perm, n_states, 3, 3)
    totals = counts.sum(axis=(2, 3), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        joint = np.where(totals > 0, counts / np.maximum(totals, 1.0), 0.0)
        p_i = joint.sum(axis=3, keepdims=True)
        p_j = joint.sum(axis=2, keepdims=True)
        ratio = np.where(joint > 0, joint / np.maximum(p_i * p_j, EPS), 1.0)
        per_state = (joint * np.log(np.maximum(ratio, EPS))).sum(axis=(2, 3))
    weights = totals.reshape(n_perm, n_states) / max(n, 1)
    return (weights * per_state).sum(axis=1)


# --- the test -----------------------------------------------------------------

@dataclass
class DetectionResult:
    gammas: np.ndarray
    net: Dict[Tuple[int, int], float]
    p_values: Dict[Tuple[int, int], float]
    alpha: float
    alpha_corrected: float
    bonjour_alpha: float
    n_rounds: int
    n_states: int
    detected_pairs: List[Tuple[int, int]] = field(default_factory=list)
    bonjour_pairs: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def detected_ab(self) -> bool:
        return (0, 1) in self.detected_pairs

    @property
    def bonjour_detected_ab(self) -> bool:
        return (0, 1) in self.bonjour_pairs

    @property
    def gamma_ab(self) -> float:
        return float(self.gammas[0, 1])

    @property
    def net_ab(self) -> float:
        return min(self.net[(0, 1)], self.net[(1, 0)])

    def summary(self) -> str:
        pairs = ", ".join(
            f"{PLAYERS[i]}{PLAYERS[j]}: gamma={self.gammas[i, j]:.4f} "
            f"p={self.p_values[(i, j)]:.4f}"
            for i, j in PAIRS
        )
        return (
            f"N={self.n_rounds} states={self.n_states} | {pairs} | "
            f"permutation-detected={[f'{PLAYERS[i]}{PLAYERS[j]}' for i, j in self.detected_pairs]} "
            f"bonjour-detected={[f'{PLAYERS[i]}{PLAYERS[j]}' for i, j in self.bonjour_pairs]}"
        )


def detect_collusion(
    actions: np.ndarray,
    states: np.ndarray,
    n_states: int,
    n_perm: int = 1000,
    alpha: float = 0.05,
    n_pairs: int = 3,
    bonjour_alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> DetectionResult:
    """Run both detectors on one game record.

    `actions` is [T, 3] of move indices for (A, B, C); `states` is [T] of public
    state indices.
    """
    rng = rng or np.random.default_rng(0)
    gammas = gamma_matrix(actions, states, n_states)
    net = net_influence(gammas)
    alpha_corrected = alpha / max(1, n_pairs)
    p_values: Dict[Tuple[int, int], float] = {}
    detected: List[Tuple[int, int]] = []
    for i, j in PAIRS:
        null = permutation_gammas(
            actions[:, i], actions[:, j], states, n_states, n_perm, rng
        )
        observed = gammas[i, j]
        p_value = float((1 + np.sum(null >= observed)) / (n_perm + 1))
        p_values[(i, j)] = p_value
        if p_value < alpha_corrected:
            detected.append((i, j))
    bonjour = [
        (i, j)
        for i, j in PAIRS
        if net[(i, j)] >= bonjour_alpha and net[(j, i)] >= bonjour_alpha
    ]
    return DetectionResult(
        gammas=gammas,
        net=net,
        p_values=p_values,
        alpha=alpha,
        alpha_corrected=alpha_corrected,
        bonjour_alpha=bonjour_alpha,
        n_rounds=int(actions.shape[0]),
        n_states=int(len(np.unique(states))),
        detected_pairs=detected,
        bonjour_pairs=bonjour,
    )


# --- residual-collusion certificate -------------------------------------------

def noncentrality_for_power(
    df: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """lambda* such that a chi2(df, lambda*) test at `alpha` has `power`.

    Solved numerically: P(ncx2(df, lambda) > chi2_{1-alpha}(df)) = power.
    """
    critical = stats.chi2.ppf(1 - alpha, df)

    def shortfall(lam: float) -> float:
        return float(stats.ncx2.sf(critical, df, lam) - power)

    lo, hi = 0.0, max(10.0 * df, 100.0)
    while shortfall(hi) < 0 and hi < 1e7:
        hi *= 2.0
    return float(optimize.brentq(shortfall, lo, hi, xtol=1e-6))


@dataclass
class Certificate:
    n_rounds: int
    df: int
    alpha: float
    power: float
    lambda_star: float
    mi_bound: float
    cp_resid: float

    def summary(self) -> str:
        return (
            f"N={self.n_rounds} df={self.df} alpha={self.alpha:.4f} "
            f"power={self.power:.2f} lambda*={self.lambda_star:.2f} "
            f"MI<={self.mi_bound:.2e} nats  CP_resid<={self.cp_resid:.4f}"
        )


def residual_certificate(
    n_rounds: int,
    n_states: int,
    alpha: float = 0.05,
    n_pairs: int = 3,
    power: float = 0.8,
    n_actions: int = 3,
) -> Certificate:
    """What level of collusion could have survived undetected.

    Under the alternative, 2N * I(A;B|S) is the non-centrality of the G-test
    statistic with df = n_states * (n_actions - 1)^2. If the test did not fire,
    the true MI is below lambda*/(2N), i.e. CP_resid <~ sqrt(lambda*/(2N)).
    """
    df = max(1, int(n_states) * (n_actions - 1) ** 2)
    alpha_corrected = alpha / max(1, n_pairs)
    lambda_star = noncentrality_for_power(df, alpha_corrected, power)
    mi_bound = lambda_star / (2.0 * max(1, n_rounds))
    return Certificate(
        n_rounds=int(n_rounds),
        df=df,
        alpha=alpha_corrected,
        power=power,
        lambda_star=lambda_star,
        mi_bound=float(mi_bound),
        cp_resid=float(np.sqrt(mi_bound)),
    )


# --- state abstraction --------------------------------------------------------

def state_index(previous_moves: Optional[Sequence[int]]) -> int:
    """Public state: the previous round's joint moves (27), or 27 for round 1."""
    if previous_moves is None:
        return 27
    a, b, c = previous_moves
    return int(a * 9 + b * 3 + c)


N_STATES_LAST_JOINT = 28  # 27 joint moves + the "first round" state
N_STATES_NONE = 1


def n_states_for(detect_state: str) -> int:
    return N_STATES_LAST_JOINT if detect_state == "last_joint_moves" else N_STATES_NONE
