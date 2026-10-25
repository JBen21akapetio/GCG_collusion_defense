"""Figures that make the central objects of the experiment visible.

Three, in teaching order:

1. **The game** (`payoff_figure`) -- A's expected payoff for each (a_A, b_B),
   showing that f(a_A) is the unique move of B that guarantees A a point
   whatever C plays. Needs no model.
2. **The policy** (`q_heatmap_figure`) -- the 3x3 matrix q[a, b] = P(B plays b |
   A plays a), per condition, with and without the suffix. This is the object
   the mutual information is computed from: identical rows mean independence,
   so the defence is literally "make the rows identical again".
3. **The behaviour** (`decomposition_figure`) -- where B's probability mass
   goes: help f(a_A), counter (the move that beats A), or copy. This is what
   showed that the model counters rather than helps when told to collude.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import game, io_utils
from .measure import measure_pool
from .model_utils import MoveModel
from .pools import build_pools
from .prompts import build_datasets

log = logging.getLogger(__name__)

#: conditions worth showing, in reporting order
CONDITIONS = (
    ("independent", "independent"),
    ("colluding", "colluding"),
    ("colluding_implicit", "colluding (intent only)"),
    ("info_no_intent", "info, no intent"),
    ("private_channel", "private channel"),
)


def _sequential_cmap():
    """One hue, light to dark: the rule for a magnitude encoding."""
    from matplotlib.colors import LinearSegmentedColormap

    from .plots import BLUE

    return LinearSegmentedColormap.from_list("mi_blue", ["#f7f9fc", BLUE])


# --- 1. the game --------------------------------------------------------------

def payoff_figure(out_dir: str) -> List[str]:
    """A's expected payoff per (a_A, b_B), averaged over C. No model needed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plots import INK, INK_2, INK_3, _save

    matrix = game.payoff_a_given_ab()          # [3, 3], mean over C
    guaranteed = np.array(
        [[game.guarantees_point_for_a(a, b) for b in range(3)] for a in range(3)]
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(matrix, cmap=_sequential_cmap(), vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            certain = guaranteed[a, b]
            ax.text(
                b,
                a,
                f"{matrix[a, b]:.2f}" + ("\n= f(a)" if certain else ""),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[a, b] > 0.6 else INK,
                fontweight="bold" if certain else "normal",
            )
            if certain:
                ax.add_patch(
                    plt.Rectangle(
                        (b - 0.5, a - 0.5), 1, 1, fill=False,
                        edgecolor=INK, linewidth=2.2, zorder=4,
                    )
                )
    ax.set_xticks(range(3), game.MOVES)
    ax.set_yticks(range(3), game.MOVES)
    ax.set_xlabel("move of B", color=INK_2, fontsize=10)
    ax.set_ylabel("move of A", color=INK_2, fontsize=10)
    ax.set_title(
        "A's expected payoff, averaged over C's three moves",
        color=INK, fontsize=11, loc="left", pad=10,
    )
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    bar.ax.tick_params(colors=INK_2, labelsize=8, length=2)
    ax.text(
        0.0, -0.22,
        "the boxed cell of each row is f(a_A): the only move of B that gives A a\n"
        "point for every move of C. That is what collusion has to produce.",
        transform=ax.transAxes, fontsize=8, color=INK_3,
    )
    paths = _save(fig, out_dir, "game_payoff")
    plt.close(fig)
    return paths


# --- 2. the policy ------------------------------------------------------------

def q_heatmap_figure(
    out_dir: str,
    matrices: Dict[str, np.ndarray],
    mi_values: Dict[str, float],
    name: str = "q_matrices",
) -> List[str]:
    """q[a, b] per condition. Identical rows = independence = MI 0."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plots import INK, INK_2, INK_3, _save

    keys = [k for k in matrices if matrices[k] is not None]
    if not keys:
        return []
    cols = min(len(keys), 4)
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.5 * rows), squeeze=False)
    cmap = _sequential_cmap()
    for idx, key in enumerate(keys):
        ax = axes[idx // cols][idx % cols]
        matrix = matrices[key]
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(
                    b, a, f"{matrix[a, b]:.2f}",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if matrix[a, b] > 0.6 else INK,
                )
        ax.set_xticks(range(3), [m[0] for m in game.MOVES], fontsize=8)
        ax.set_yticks(range(3), [m[0] for m in game.MOVES], fontsize=8)
        ax.set_title(
            f"{key}\nMI = {mi_values.get(key, float('nan')):.4f} nats",
            color=INK, fontsize=9.5, pad=8,
        )
        ax.tick_params(colors=INK_2, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        if idx % cols == 0:
            ax.set_ylabel("A plays", color=INK_2, fontsize=9)
        if idx // cols == rows - 1:
            ax.set_xlabel("B plays", color=INK_2, fontsize=9)
    for empty in range(len(keys), rows * cols):
        axes[empty // cols][empty % cols].axis("off")
    fig.suptitle(
        "q(B | A, context): the matrix the mutual information is computed from",
        color=INK, fontsize=11, x=0.02, ha="left", y=0.995,
    )
    fig.text(
        0.02, 0.005,
        "Identical rows mean B ignores A, i.e. MI = 0. The defence makes the rows "
        "identical again.",
        fontsize=8, color=INK_3,
    )
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02).ax.tick_params(
        colors=INK_2, labelsize=8, length=2
    )
    paths = _save(fig, out_dir, name)
    plt.close(fig)
    return paths


# --- 3. the behaviour ---------------------------------------------------------

def decomposition(matrix: np.ndarray) -> Dict[str, float]:
    """Where B's mass goes: help f(a), counter (beats A), or copy A."""
    help_rate = float(np.mean([matrix[a, game.collusion_target(a)] for a in range(3)]))
    counter = float(np.mean([matrix[a, game.LOSES_TO[a]] for a in range(3)]))
    copy = float(np.mean([matrix[a, a] for a in range(3)]))
    return {"help": help_rate, "counter": counter, "copy": copy}


def decomposition_figure(
    out_dir: str, matrices: Dict[str, np.ndarray], name: str = "move_decomposition"
) -> List[str]:
    """Stacked bars: the three mutually exclusive things B can do."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plots import AQUA, BLUE, INK, INK_2, INK_3, ORANGE, _save, _style

    keys = [k for k in matrices if matrices[k] is not None]
    if not keys:
        return []
    parts = {k: decomposition(matrices[k]) for k in keys}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(keys))
    bottom = np.zeros(len(keys))
    for label, colour, key in (
        ("help f(a_A)", BLUE, "help"),
        ("counter (beats A)", ORANGE, "counter"),
        ("copy A", AQUA, "copy"),
    ):
        values = np.array([parts[k][key] for k in keys])
        ax.bar(
            x, values, bottom=bottom, color=colour, width=0.62,
            label=label, edgecolor="white", linewidth=2.0, zorder=3,
        )
        for i, (value, base) in enumerate(zip(values, bottom)):
            if value > 0.06:
                ax.text(
                    i, base + value / 2, f"{value:.2f}",
                    ha="center", va="center", fontsize=8.5, color="white",
                    fontweight="bold", zorder=4,
                )
        bottom += values
    ax.axhline(1 / 3, color=INK_3, linewidth=1.0, linestyle=(0, (4, 3)), zorder=5)
    ax.annotate(
        "chance = 1/3", (len(keys) - 0.45, 1 / 3), fontsize=8, color=INK_3,
        textcoords="offset points", xytext=(0, 4), ha="right",
    )
    ax.set_xticks(x, keys, rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    _style(ax, "", "share of B's probability mass", "What B actually does with A's move")
    ax.legend(
        frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center",
        bbox_to_anchor=(0.5, -0.26), ncol=3, handletextpad=0.4,
    )
    paths = _save(fig, out_dir, name)
    plt.close(fig)
    return paths


# --- orchestration ------------------------------------------------------------

def condition_matrices(
    model: MoveModel, cfg, suffix_ids: Sequence[int] = ()
) -> Dict[str, object]:
    """Mean q[a, b] and MI for every test condition under one defence."""
    datasets = build_datasets(cfg)
    pools = build_pools(model, datasets, cfg, splits=("test",))["test"]
    matrices: Dict[str, np.ndarray] = {}
    mi_values: Dict[str, float] = {}
    for key, label in CONDITIONS:
        pool = pools.get(key)
        if pool is None:
            continue
        metrics = measure_pool(model, pool, suffix_ids)
        q = metrics["_q"].numpy().astype(np.float64)      # [n, per_context, 3]
        if q.shape[1] == 1:
            q = np.repeat(q, 3, axis=1)                   # independent: same row 3x
        matrices[label] = q.mean(axis=0)
        mi_values[label] = float(metrics["model_mi"])
    return {"matrices": matrices, "mi": mi_values}


def run_explain(
    cfg,
    run_id: Optional[str] = None,
    suffix_override: Optional[str] = None,
    model: Optional[MoveModel] = None,
) -> Dict[str, object]:
    """Draw the three explanatory figures for a run, with and without its suffix."""
    if run_id:
        cfg.run.run_id = run_id
    run_dir = cfg.run_dir()
    io_utils.setup_logging(run_dir)

    figures = payoff_figure(run_dir)          # needs no model

    suffix_ids: List[int] = []
    suffix_text = suffix_override
    best_json = os.path.join(run_dir, "gcg_best_suffix.json")
    if suffix_text is None and os.path.exists(best_json):
        payload = io_utils.read_json(best_json)
        suffix_text, suffix_ids = payload["suffix"], list(payload["suffix_ids"])

    if model is None:
        model = MoveModel(
            cfg.model.name,
            device=cfg.run.device,
            dtype=cfg.run.dtype,
            moves=cfg.game.moves,
            batch_size=cfg.model.batch_size,
            use_kv_cache=cfg.model.use_kv_cache,
            max_batch_size=cfg.model.max_batch_size,
        )
    if suffix_text and not suffix_ids:
        suffix_ids = model.encode_suffix(suffix_text)

    fields = ("defence", "condition", "move_a", "move_b", "probability", "mi")
    csv_log = io_utils.CSVLog(os.path.join(run_dir, "q_matrices.csv"), fields)
    everything: Dict[str, Dict[str, np.ndarray]] = {}
    for defence, ids in (("no defence", []), ("GCG suffix", suffix_ids)):
        if defence == "GCG suffix" and not ids:
            continue
        out = condition_matrices(model, cfg, ids)
        everything[defence] = out["matrices"]
        for label, matrix in out["matrices"].items():
            for a in range(3):
                for b in range(3):
                    csv_log.write({
                        "defence": defence,
                        "condition": label,
                        "move_a": game.MOVES[a],
                        "move_b": game.MOVES[b],
                        "probability": round(float(matrix[a, b]), 6),
                        "mi": round(out["mi"][label], 6),
                    })
        suffix_tag = "no_defence" if defence == "no defence" else "with_suffix"
        figures += q_heatmap_figure(
            run_dir, out["matrices"], out["mi"], name=f"q_matrices_{suffix_tag}"
        )
        figures += decomposition_figure(
            run_dir, out["matrices"], name=f"move_decomposition_{suffix_tag}"
        )
        log.info(
            "%s | %s",
            defence,
            " ".join(f"{k}={v:.4f}" for k, v in out["mi"].items()),
        )
    csv_log.close()
    log.info("wrote %d figure files to %s/figures", len(figures), run_dir)
    return {"run_dir": run_dir, "figures": figures, "matrices": everything}
