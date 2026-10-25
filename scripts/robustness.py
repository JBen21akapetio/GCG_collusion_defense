#!/usr/bin/env python
"""Robustness studies for a found suffix.

    # 1. perturbations : le suffixe est-il un point isolé ou une région ?
    python scripts/robustness.py perturb --run-id <run_id>

    # 2 & 3. comparer les suffixes de plusieurs runs (modèles ou attaquants)
    python scripts/robustness.py compare --run-ids <id1>,<id2>,<id3>

`compare` est purement textuel : il ne charge aucun modèle et ne coûte rien.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gcg_defense import io_utils  # noqa: E402
from gcg_defense.config import Config  # noqa: E402
from gcg_defense.robustness import compare_runs, run_perturbation_study  # noqa: E402


def cmd_perturb(args) -> int:
    config = args.config
    saved = os.path.join("results", args.run_id, "config.yaml")
    if os.path.exists(saved):
        config = saved
    cfg = Config.load(config, args.overrides)
    out = run_perturbation_study(
        cfg, run_id=args.run_id, suffix_override=args.suffix, draws=args.draws
    )
    print(out["summary"])
    return 0


def cmd_compare(args) -> int:
    io_utils.setup_logging(None)
    run_dirs = [
        d if os.path.sep in d else os.path.join("results", d)
        for d in args.run_ids.split(",")
        if d.strip()
    ]
    print(compare_runs(run_dirs))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    perturb = sub.add_parser("perturb", help="score les variantes du suffixe")
    perturb.add_argument("--run-id", required=True)
    perturb.add_argument("--config", default="configs/default.yaml")
    perturb.add_argument("--set", dest="overrides", action="append", default=[])
    perturb.add_argument("--suffix", default=None, help="suffixe explicite au lieu du run")
    perturb.add_argument("--draws", type=int, default=3)
    perturb.set_defaults(func=cmd_perturb)

    compare = sub.add_parser("compare", help="compare les suffixes de plusieurs runs")
    compare.add_argument("--run-ids", required=True, help="liste séparée par des virgules")
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
