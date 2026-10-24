#!/usr/bin/env python
"""Benign optimiser control: force B to answer "Paper" on independent prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gcg_defense.config import Config  # noqa: E402
from gcg_defense.positive_control import run_positive_control  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.overrides)
    payload = run_positive_control(cfg, n_steps=args.steps)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
