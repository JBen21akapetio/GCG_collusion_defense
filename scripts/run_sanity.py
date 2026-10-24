#!/usr/bin/env python
"""Run the pre-optimisation sanity checks and exit non-zero if any fails."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gcg_defense import io_utils  # noqa: E402
from gcg_defense.config import Config  # noqa: E402
from gcg_defense.sanity import run_sanity  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.overrides)
    run_dir = cfg.run_dir()
    io_utils.setup_logging(run_dir)
    cfg.save(run_dir)
    report = run_sanity(cfg, run_dir=run_dir)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
