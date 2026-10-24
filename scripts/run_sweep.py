#!/usr/bin/env python
"""Sweep seeds and hyper-parameters, then draw the combined Pareto frontier.

    python scripts/run_sweep.py --config configs/default.yaml \
        --seeds 0 1 2 --lambda-util 0.1 1 10

Each cell is a full GCG run plus its evaluation; the figures at the end carry
error bars over seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gcg_defense import io_utils, plots  # noqa: E402
from gcg_defense.config import Config  # noqa: E402
from gcg_defense.evaluate import run_evaluation  # noqa: E402
from gcg_defense.gcg import run_gcg  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--lambda-util", type=float, nargs="+", default=None)
    parser.add_argument("--lambda-repr", type=float, nargs="+", default=None)
    parser.add_argument("--suffix-len", type=int, nargs="+", default=None)
    parser.add_argument("--suffix-position", nargs="+", default=None)
    parser.add_argument("--out", default=None, help="directory for the merged figures")
    args = parser.parse_args(argv)

    base = Config.load(args.config, args.overrides)
    seeds = args.seeds or base.run.seeds
    grid = {
        "loss.lambda_util": args.lambda_util or [base.loss.lambda_util],
        "loss.lambda_repr": args.lambda_repr or [base.loss.lambda_repr],
        "gcg.suffix_len": args.suffix_len or [base.gcg.suffix_len],
        "prompts.suffix_position": args.suffix_position or [base.prompts.suffix_position],
    }
    keys = list(grid)
    cells = list(itertools.product(*(grid[k] for k in keys)))
    io_utils.setup_logging(None)
    print(f"sweep: {len(seeds)} seeds x {len(cells)} cells = {len(seeds) * len(cells)} runs")

    metrics_csvs, results = [], []
    for cell in cells:
        for seed in seeds:
            overrides = list(args.overrides)
            overrides.append(f"run.seed={seed}")
            tag = "_".join(
                f"{k.split('.')[-1]}{v}".replace(".", "p") for k, v in zip(keys, cell)
            )
            overrides.append(f"run.tag={tag}")
            overrides.extend(f"{k}={v}" for k, v in zip(keys, cell))
            cfg = Config.load(args.config, overrides)
            out = run_gcg(cfg, skip_sanity=True)
            cfg_eval = Config.load(args.config, overrides)
            cfg_eval.run.run_id = out["run_id"]
            evaluated = run_evaluation(cfg_eval, run_id=out["run_id"])
            metrics_csvs.append(os.path.join(evaluated["run_dir"], "metrics.csv"))
            results.append(
                {"run_id": out["run_id"], "seed": seed, **dict(zip(keys, cell))}
            )

    out_dir = args.out or os.path.join(base.run.out_dir, "sweep")
    os.makedirs(out_dir, exist_ok=True)
    figures = plots.pareto(out_dir, metrics_csvs, base)
    figures += plots.detection_curves(out_dir, metrics_csvs)
    io_utils.write_json(
        os.path.join(out_dir, "sweep.json"),
        {"runs": results, "metrics_csvs": metrics_csvs, "figures": figures},
    )
    print(json.dumps({"out_dir": out_dir, "figures": figures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
