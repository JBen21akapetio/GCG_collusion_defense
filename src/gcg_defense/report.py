"""Turn `metrics.csv` into the markdown tables used in the write-up."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

from . import io_utils
from .plots import load_metrics

MAIN_ROWS = (
    ("model_mi", "colluding", "test", "model-MI"),
    ("help_rate", "colluding", "test", "help rate"),
    ("util_kl", "independent", "test", "utility KL"),
    ("validity", "independent", "test", "validity"),
    ("entropy", "independent", "test", "H(q) indep"),
    ("best_response_gap", "independent", "test", "exploitability"),
)

GENERALISATION = (
    ("colluding", "held-out instructions"),
    ("colluding_train_instr", "train instructions"),
    ("colluding_implicit", "intent-only wordings"),
    ("info_no_intent", "info, no intent"),
    ("private_channel", "private channel"),
    ("independent", "independent"),
)


def _pivot(frame, metric: str, condition: str, split: str = "test"):
    sub = frame[
        (frame["split"] == split)
        & (frame["condition"] == condition)
        & (frame["metric"] == metric)
    ]
    return sub.groupby("method")["value"].mean()


def _fmt(value, places: int = 4) -> str:
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "--"


def build_report(run_dir: str, csv_paths: Optional[Sequence[str]] = None) -> str:
    """Markdown tables: headline metrics, generalisation, behavioural test."""
    paths = list(csv_paths or [os.path.join(run_dir, "metrics.csv")])
    frame = load_metrics(paths)
    if frame is None:
        return "no metrics found\n"
    methods = sorted(
        frame[frame["split"] == "test"]["method"].unique(),
        key=lambda m: (m != "none", m != "gcg_suffix", m),
    )
    lines: List[str] = ["## Headline metrics (held-out contexts and instructions)", ""]
    header = "| method | " + " | ".join(label for *_, label in MAIN_ROWS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(MAIN_ROWS) + 1))
    series = {
        label: _pivot(frame, metric, condition, split)
        for metric, condition, split, label in MAIN_ROWS
    }
    for method in methods:
        cells = [_fmt(series[label].get(method)) for *_, label in MAIN_ROWS]
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")

    lines += ["", "## Generalisation: model-MI by condition (nats)", ""]
    lines.append("| method | " + " | ".join(label for _, label in GENERALISATION) + " |")
    lines.append("|" + "---|" * (len(GENERALISATION) + 1))
    mi_by_condition = {
        condition: _pivot(frame, "model_mi", condition) for condition, _ in GENERALISATION
    }
    for method in methods:
        cells = [_fmt(mi_by_condition[c].get(method)) for c, _ in GENERALISATION]
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")

    sim = frame[frame["split"] == "test_sim"].copy()
    if not sim.empty:
        # Older runs label the condition "n_rounds=N" with no state prefix.
        sim["n_rounds"] = sim["condition"].str.extract(r"n_rounds=(\d+)")
        sim = sim.dropna(subset=["n_rounds"])
        sim["n_rounds"] = sim["n_rounds"].astype(int)
        sim["state"] = (
            sim["condition"].str.extract(r"state=([a-z_]+)")[0].fillna("unspecified")
        )
        for state in sorted(sim["state"].unique()):
            lines += ["", f"## Behavioural test (state = {state})", ""]
            rounds = sorted(sim["n_rounds"].unique())
            lines.append(
                "| method | "
                + " | ".join(f"detect @ N={n}" for n in rounds)
                + " | A-C win rate @ N="
                + str(max(rounds))
                + " |"
            )
            lines.append("|" + "---|" * (len(rounds) + 2))
            for method in methods:
                cells = []
                for n in rounds:
                    value = sim[
                        (sim["method"] == method)
                        & (sim["state"] == state)
                        & (sim["n_rounds"] == n)
                        & (sim["metric"] == "detection_rate")
                    ]["value"]
                    cells.append(_fmt(value.mean() if not value.empty else None, 3))
                gap = sim[
                    (sim["method"] == method)
                    & (sim["state"] == state)
                    & (sim["n_rounds"] == max(rounds))
                    & (sim["metric"] == "a_minus_c_win_rate")
                ]["value"]
                cells.append(_fmt(gap.mean() if not gap.empty else None, 4))
                lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
            certificate = sim[
                (sim["state"] == state) & (sim["metric"] == "certificate_cp_resid")
            ]
            if not certificate.empty:
                bounds = (
                    certificate.groupby("n_rounds")["value"].first().sort_index()
                )
                lines += [
                    "",
                    "Residual-collusion certificate (a property of the test, not of a "
                    "method): "
                    + ", ".join(f"N={n}: CP_resid <= {v:.3f}" for n, v in bounds.items()),
                ]

    transfer = frame[frame["split"] == "test_transfer"]
    if not transfer.empty:
        lines += ["", "## Transfer (suffix text re-encoded per tokeniser)", ""]
        lines.append("| model | method | model-MI | help rate | utility KL |")
        lines.append("|---|---|---|---|---|")
        for model_name in sorted(transfer["model"].unique()):
            for method in ("none", "gcg_suffix"):
                rows = transfer[
                    (transfer["model"] == model_name) & (transfer["method"] == method)
                ]
                mi = rows[(rows["condition"] == "colluding") & (rows["metric"] == "model_mi")]
                help_rate = rows[
                    (rows["condition"] == "colluding") & (rows["metric"] == "help_rate")
                ]
                kl = rows[
                    (rows["condition"] == "independent") & (rows["metric"] == "util_kl")
                ]
                if mi.empty:
                    continue
                lines.append(
                    f"| `{model_name}` | `{method}` | {_fmt(mi['value'].mean())} | "
                    f"{_fmt(help_rate['value'].mean(), 3)} | {_fmt(kl['value'].mean())} |"
                )
    return "\n".join(lines) + "\n"


def write_report(run_dir: str, csv_paths: Optional[Sequence[str]] = None) -> str:
    text = build_report(run_dir, csv_paths)
    path = os.path.join(run_dir, "report.md")
    io_utils.write_text(path, text)
    return path
