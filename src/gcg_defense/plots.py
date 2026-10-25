"""Figures: the model-MI / utility Pareto frontier, GCG loss curves, detection.

Colour follows the method FAMILY, not the method's rank, and the three families
use categorical slots 1-3 only -- the slots that validate on the all-pairs
list that scatter plots require. "No defence" is drawn in neutral ink as a
reference, not as a fourth hue. Marker shape repeats the family (so identity is
never colour-alone) and the points that carry the argument are labelled
directly. One y-axis per figure, in nats; no dual axes.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# categorical slots 1-3 of the reference palette (light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"

FAMILIES: Tuple[Tuple[str, str, str, str], ...] = (
    # (family, colour, marker, legend label)
    ("gcg", BLUE, "o", "GCG suffix"),
    ("suffix_baseline", ORANGE, "s", "hand-written / random suffix"),
    ("activation", AQUA, "^", "activation intervention"),
    ("none", INK_2, "D", "no defence"),
)

#: Only the points that carry the argument are labelled; the legend names the
#: families, so labelling every member just creates collisions.
LABEL_METHODS = ("none", "gcg_suffix", "directional_ablation")


def style_of(family: str) -> Tuple[str, str, str]:
    """(colour, marker, legend label) for a method family."""
    for name, colour, marker, label in FAMILIES:
        if name == family:
            return colour, marker, label
    return INK_2, "o", family


def family_of(method: str) -> str:
    if method == "none":
        return "none"
    if method.startswith("gcg"):
        return "gcg"
    if method.startswith(("handwritten", "random_suffix")):
        return "suffix_baseline"
    if method.startswith(("directional_ablation", "steering")):
        return "activation"
    return "suffix_baseline"


def _style(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)


def _save(fig, run_dir: str, name: str) -> List[str]:
    out = []
    directory = os.path.join(run_dir, "figures")
    os.makedirs(directory, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(directory, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        out.append(path)
    return out


# --- data loading -------------------------------------------------------------

def load_metrics(paths: Sequence[str]):
    """Long-format metrics.csv files -> one DataFrame."""
    import pandas as pd

    frames = []
    for path in paths:
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        frame["source"] = path
        frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def pareto_table(frame):
    """One row per (method, seed): model-MI on colluding test vs utility KL."""
    import pandas as pd

    rows = []
    sub = frame[(frame["split"] == "test")]
    for (method, params, seed), group in sub.groupby(["method", "method_params", "seed"]):
        mi = group[(group["condition"] == "colluding") & (group["metric"] == "model_mi")]
        mi_sem = group[
            (group["condition"] == "colluding") & (group["metric"] == "model_mi_sem")
        ]
        kl = group[(group["condition"] == "independent") & (group["metric"] == "util_kl")]
        kl_sem = group[
            (group["condition"] == "independent") & (group["metric"] == "util_kl_sem")
        ]
        flag = group[(group["condition"] == "flags") & (group["metric"] == "constant_move_flag")]
        if mi.empty or kl.empty:
            continue
        rows.append(
            {
                "method": method,
                "params": params,
                "seed": seed,
                "model_mi": float(mi["value"].iloc[0]),
                "model_mi_sem": float(mi_sem["value"].iloc[0]) if not mi_sem.empty else 0.0,
                "util_kl": float(kl["value"].iloc[0]),
                "util_kl_sem": float(kl_sem["value"].iloc[0]) if not kl_sem.empty else 0.0,
                "constant_move_flag": float(flag["value"].iloc[0]) if not flag.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


# --- figures ------------------------------------------------------------------

def pareto(run_dir: str, csv_paths: Sequence[str], cfg=None) -> List[str]:
    """Main figure: model-MI (y) against utility KL (x), one point per method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as patheffects
    import matplotlib.pyplot as plt

    frame = load_metrics(csv_paths)
    if frame is None:
        return []
    table = pareto_table(frame)
    if table.empty:
        return []
    # Error bars over seeds when several are present, else the SEM over contexts.
    grouped = table.groupby(["method", "params"], as_index=False)
    agg = grouped.agg(
        model_mi=("model_mi", "mean"),
        util_kl=("util_kl", "mean"),
        mi_seed_sd=("model_mi", "std"),
        kl_seed_sd=("util_kl", "std"),
        mi_sem=("model_mi_sem", "mean"),
        kl_sem=("util_kl_sem", "mean"),
        n_seeds=("seed", "nunique"),
        flag=("constant_move_flag", "max"),
    )
    over_seeds = bool((agg["n_seeds"] > 1).any())
    agg["mi_err"] = agg["mi_seed_sd"].fillna(0.0) if over_seeds else agg["mi_sem"]
    agg["kl_err"] = agg["kl_seed_sd"].fillna(0.0) if over_seeds else agg["kl_sem"]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    # Labels go left of a point only in the right half of the data range, so a
    # label never runs back across the crowded low-cost cluster.
    x_midpoint = (agg["util_kl"].min() + agg["util_kl"].max()) / 2
    seen = set()
    label_count = 0
    for _, row in agg.iterrows():
        family = family_of(row["method"])
        colour, marker, label = style_of(family)
        ax.errorbar(
            row["util_kl"],
            row["model_mi"],
            xerr=row["kl_err"],
            yerr=row["mi_err"],
            fmt=marker,
            markersize=8,
            markerfacecolor=colour,
            markeredgecolor="white",
            markeredgewidth=1.2,
            ecolor=colour,
            elinewidth=1.2,
            capsize=2.5,
            alpha=0.95,
            zorder=3,
            label=label if label not in seen else None,
        )
        seen.add(label)
        if row["method"] in LABEL_METHODS or row["flag"] > 0:
            text = row["method"] + (" (constant!)" if row["flag"] > 0 else "")
            right_half = row["util_kl"] > x_midpoint
            dy = 7 if label_count % 2 == 0 else -13
            label_count += 1
            ax.annotate(
                text,
                (row["util_kl"], row["model_mi"]),
                textcoords="offset points",
                xytext=(-10 if right_half else 10, dy),
                ha="right" if right_half else "left",
                fontsize=8,
                color=INK_2,
                zorder=5,
                path_effects=[
                    patheffects.withStroke(linewidth=2.5, foreground="white")
                ],
            )
    baseline = agg[agg["method"] == "none"]
    if not baseline.empty:
        ax.axhline(
            float(baseline["model_mi"].iloc[0]),
            color=INK_3,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
    err_note = "error bars: SD over seeds" if over_seeds else "error bars: SEM over contexts"
    _style(
        ax,
        "utility cost: KL on independent test contexts (nats)",
        "model-MI I(A;B | context) on colluding test contexts (nats)",
        "Removing collusion versus the cost to legitimate play",
    )
    # Legend below the axes: the upper right of this plot is where the worst
    # methods land, so an in-axes legend would sit on top of them.
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    ax.text(
        0.0,
        -0.26,
        f"lower is better on both axes; dashed line = no defence; {err_note}",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_3,
    )
    paths = _save(fig, run_dir, "pareto_mi_vs_utility")
    plt.close(fig)
    return paths


def _load_evals(run_dir: str, label: str):
    """Full-pool evaluations: from their CSV, or from the checkpoint if a run
    pre-dates that CSV."""
    import pandas as pd

    path = os.path.join(run_dir, f"{label}_evals.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    checkpoint = os.path.join(run_dir, f"{label}_checkpoint.pt")
    if not os.path.exists(checkpoint):
        return None
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return pd.DataFrame(payload.get("evals", []))
    except Exception as exc:  # pragma: no cover
        log.warning("could not read evals from %s: %s", checkpoint, exc)
        return None


def loss_curves(run_dir: str, label: str = "gcg") -> List[str]:
    """GCG loss curves: the minibatch loss per step and the full-pool evals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.ticker import MaxNLocator

    steps_path = os.path.join(run_dir, f"{label}_steps.csv")
    if not os.path.exists(steps_path):
        return []
    steps = pd.read_csv(steps_path)
    if steps.empty:
        return []
    evals = _load_evals(run_dir, label)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(
        steps["step"],
        steps["loss"],
        color=INK_3,
        linewidth=1.0,
        alpha=0.8,
        zorder=2,
        label="total loss (minibatch)",
    )
    if "mi" in steps and steps["mi"].notna().any():
        ax.plot(
            steps["step"],
            steps["mi"],
            color=BLUE,
            linewidth=1.8,
            zorder=3,
            label="MI term",
        )
    if "util" in steps and steps["util"].notna().any():
        ax.plot(
            steps["step"],
            steps["util"],
            color=ORANGE,
            linewidth=1.8,
            zorder=3,
            label="utility term",
        )
    if evals is not None and not evals.empty:
        # The minibatch loss resamples 8 contexts every step, so it is noisy by
        # construction; the full-pool evaluations are the real trajectory.
        ax.plot(
            evals["step"],
            evals["loss"],
            color=INK,
            linewidth=2.0,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
            label="total loss (full train pool)",
        )
    _style(
        ax,
        "GCG step",
        "loss (nats)",
        "GCG optimisation of the defensive suffix",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    paths = _save(fig, run_dir, f"{label}_loss_curve")
    plt.close(fig)
    return paths


def detection_curves(
    run_dir: str, csv_paths: Sequence[str], state: Optional[str] = None
) -> List[str]:
    """Detection rate of the calibrated test against the number of rounds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = load_metrics(csv_paths)
    if frame is None:
        return []
    sub = frame[(frame["split"] == "test_sim") & (frame["metric"] == "detection_rate")]
    if sub.empty:
        return []
    sub = sub.copy()
    sub["n_rounds"] = sub["condition"].str.extract(r"n_rounds=(\d+)").astype(int)
    sub["detect_state"] = sub["condition"].str.extract(r"state=([a-z_]+)")
    primary = state or "last_joint_moves"
    if (sub["detect_state"] == primary).any():
        sub = sub[sub["detect_state"] == primary]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sub["family"] = sub["method"].map(family_of)
    # Several methods of a family land on exactly the same value (often 1.0),
    # so plot the family mean with a min-max band rather than one line per
    # method, and draw the reference dashed ON TOP so overlapping families
    # stay visible through the gaps.
    order = [f for f, *_ in FAMILIES if f != "none"] + ["none"]
    for family in order:
        rows = sub[sub["family"] == family]
        if rows.empty:
            continue
        colour, marker, label = style_of(family)
        stats = rows.groupby("n_rounds")["value"].agg(["mean", "min", "max"]).reset_index()
        reference = family == "none"
        if (stats["max"] - stats["min"]).max() > 1e-9:
            ax.fill_between(
                stats["n_rounds"],
                stats["min"],
                stats["max"],
                color=colour,
                alpha=0.15,
                linewidth=0,
                zorder=2,
            )
        ax.plot(
            stats["n_rounds"],
            stats["mean"],
            color=colour,
            marker=marker,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0,
            linestyle=(0, (5, 2)) if reference else "-",
            zorder=6 if reference else 3,
            label=label,
        )
    ax.axhline(0.05, color=INK_3, linewidth=0.9, linestyle=(0, (2, 3)), zorder=1)
    ax.set_ylim(-0.05, 1.08)
    _style(
        ax,
        "rounds per game (N)",
        "detection rate of the A-B pair",
        f"Behavioural detection: permutation test, state = {primary}",
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    ax.text(
        0.0,
        -0.28,
        "shaded band = range within the family; dotted line = the 0.05 level the "
        "test is calibrated to",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_3,
    )
    paths = _save(fig, run_dir, f"detection_rate_{primary}")
    plt.close(fig)
    return paths


def make_all(run_dir: str, cfg=None, extra_csvs: Sequence[str] = ()) -> List[str]:
    """Every figure for one run (plus any extra metrics.csv for seed error bars)."""
    csv_paths = [os.path.join(run_dir, "metrics.csv"), *extra_csvs]
    figures: List[str] = []
    try:
        figures += pareto(run_dir, csv_paths, cfg)
        figures += loss_curves(run_dir, "gcg")
        figures += loss_curves(run_dir, "positive_control")
        figures += detection_curves(run_dir, csv_paths, "last_joint_moves")
        figures += detection_curves(run_dir, csv_paths, "none")
    except Exception as exc:  # pragma: no cover - never fail a run over a figure
        log.warning("figure generation failed: %s", exc)
    if figures:
        log.info("wrote %d figure files to %s/figures", len(figures), run_dir)
    return figures


#: Perturbation families. Two hues plus neutral ink for the controls keeps the
#: scatter within the all-pairs colour limit.
PERTURBATION_FAMILIES = (
    ("token", BLUE, "o", "token perturbation"),
    ("char", ORANGE, "s", "character edit"),
    ("control", INK_2, "D", "control (random / empty)"),
)

_TOKEN_KINDS = ("replace", "delete", "shuffle", "reverse", "truncate")


def perturbation_family(kind: str) -> str:
    if kind in ("random", "none"):
        return "control"
    if kind == "char_edit":
        return "char"
    if any(kind.startswith(prefix) for prefix in _TOKEN_KINDS):
        return "token"
    return "control"


def robustness_scatter(run_dir: str) -> List[str]:
    """Model-MI against how close a perturbed suffix stays to the original."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    path = os.path.join(run_dir, "robustness_perturbation.csv")
    if not os.path.exists(path):
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    original = frame[frame["kind"] == "original"]
    none_row = frame[frame["kind"] == "none"]
    body = frame[~frame["kind"].isin(["original"])].copy()
    body["family"] = body["kind"].map(perturbation_family)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    seen = set()
    for family, colour, marker, label in PERTURBATION_FAMILIES:
        rows = body[body["family"] == family]
        if rows.empty:
            continue
        ax.scatter(
            rows["char_ratio"],
            rows["model_mi"],
            s=62,
            marker=marker,
            facecolor=colour,
            edgecolor="white",
            linewidth=1.1,
            alpha=0.9,
            zorder=3,
            label=label if label not in seen else None,
        )
        seen.add(label)
    if not original.empty:
        value = float(original["model_mi"].iloc[0])
        ax.axhline(value, color=BLUE, linewidth=1.6, zorder=2)
        ax.annotate(
            f"intact suffix ({value:.3f})",
            (0.015, value),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, 6),
            fontsize=8,
            color=BLUE,
        )
    if not none_row.empty:
        value = float(none_row["model_mi"].iloc[0])
        ax.axhline(value, color=INK_3, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(
            f"no defence ({value:.3f})",
            (0.015, value),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, -12),
            fontsize=8,
            color=INK_3,
        )
    _style(
        ax,
        "character similarity to the original suffix",
        "model-MI I(A;B | context) on test contexts (nats)",
        "Is the suffix an isolated point or a region?",
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        handletextpad=0.4,
    )
    ax.text(
        0.0,
        -0.26,
        "a point near the blue line keeps the defence; near the grey line it "
        "destroys it",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_3,
    )
    paths = _save(fig, run_dir, "robustness_perturbation")
    plt.close(fig)
    return paths


def decoding_curve(run_dir: str) -> List[str]:
    """Model-MI of the deployed policy against decoding temperature.

    Two series only (no defence / GCG suffix), so the neutral ink carries the
    reference and a single hue carries the defence.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    path = os.path.join(run_dir, "out_of_loop_decoding.csv")
    if not os.path.exists(path):
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    styles = {"no defence": (INK_2, "D", (0, (5, 2))), "GCG suffix": (BLUE, "o", "-")}
    for defence, group in frame.groupby("defence"):
        colour, marker, dash = styles.get(str(defence), (AQUA, "s", "-"))
        group = group.sort_values("temperature")
        for ax, column, label in (
            (ax1, "model_mi", "model-MI (nats)"),
            (ax2, "help_rate", "help rate"),
        ):
            ax.plot(
                group["temperature"], group[column], color=colour, marker=marker,
                markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                linewidth=2.0, linestyle=dash, zorder=3, label=str(defence),
            )
    for ax, label, title in (
        (ax1, "model-MI of the deployed policy (nats)", "Does the defence survive the decoder?"),
        (ax2, "help rate of the deployed policy", "What B actually plays"),
    ):
        ax.axvline(0.0, color=INK_3, linewidth=0.9, linestyle=(0, (2, 3)), zorder=1)
        ax.annotate(
            "greedy", (0.0, ax.get_ylim()[1]), xytext=(4, -12),
            textcoords="offset points", fontsize=8, color=INK_3,
        )
        _style(ax, "decoding temperature", label, title)
    ax2.axhline(1 / 3, color=INK_3, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax2.annotate(
        "chance", (1.5, 1 / 3), xytext=(-4, 5), ha="right",
        textcoords="offset points", fontsize=8, color=INK_3,
    )
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.text(
        0.0, -0.04,
        "T = 1 is the distribution the suffix was optimised on; every other point "
        "is out of that loop.",
        fontsize=8, color=INK_3,
    )
    paths = _save(fig, run_dir, "out_of_loop_decoding")
    plt.close(fig)
    return paths
