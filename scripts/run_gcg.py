#!/usr/bin/env python
"""Optimise the defensive suffix (sanity checks first, resumable)."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gcg_defense.config import Config  # noqa: E402
from gcg_defense.gcg import run_gcg  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        help="continue the run with this id from its last checkpoint",
    )
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config
    if args.resume:
        saved = os.path.join("results", args.resume, "config.yaml")
        if os.path.exists(saved):
            config_path = saved
    cfg = Config.load(config_path, args.overrides)
    out = run_gcg(cfg, resume_run_id=args.resume, skip_sanity=args.skip_sanity)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
