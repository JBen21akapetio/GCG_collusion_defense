#!/usr/bin/env python
"""Diff-in-means collusion direction r_hat (Arditi et al. 2024, section 3)."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gcg_defense import io_utils  # noqa: E402
from gcg_defense.config import Config  # noqa: E402
from gcg_defense.direction import compute_collusion_direction  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.overrides)
    io_utils.setup_logging(None)
    print(json.dumps(compute_collusion_direction(cfg, out_path=args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
