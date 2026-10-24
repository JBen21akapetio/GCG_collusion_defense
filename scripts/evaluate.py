#!/usr/bin/env python
"""Evaluate the GCG suffix and every baseline on the held-out pools."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gcg_defense.config import Config  # noqa: E402
from gcg_defense.evaluate import run_evaluation  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-id", default=None, help="run directory holding the suffix")
    parser.add_argument("--suffix", default=None, help="evaluate this suffix text instead")
    args = parser.parse_args(argv)

    config_path = args.config
    if args.run_id:
        saved = os.path.join("results", args.run_id, "config.yaml")
        if os.path.exists(saved):
            config_path = saved
    cfg = Config.load(config_path, args.overrides)
    out = run_evaluation(cfg, run_id=args.run_id, suffix_override=args.suffix)
    print(json.dumps({"run_dir": out["run_dir"], "figures": out["figures"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
