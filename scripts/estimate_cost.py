#!/usr/bin/env python
"""Estimate the duration and cost of GCG episodes on Modal, from measurements.

Three steps, in order. This script NEVER launches anything: step 1 prints the
commands for you to run.

  1. benchmark   print the short benchmark command for each candidate GPU
                 python scripts/estimate_cost.py benchmark --gpus A10G,L4,A100

  2. compare     turn the benchmark steps.csv files into a speed-per-dollar table
                 python scripts/estimate_cost.py compare \
                     --measured A10G=results/<id>/gcg_steps.csv \
                     --measured A100=results/<id>/gcg_steps.csv

  3. plan        cost the real sweep with the measured s/step
                 python scripts/estimate_cost.py plan --gpu A10G --s-per-step 12.4 \
                     --seeds 0,1,2 --lambda-util 0.1,1,10 --max-concurrent 4

Prices come from configs/pricing.json, which records the page they were read
from and the date. Re-fetch before trusting a number.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gcg_defense.cost import (  # noqa: E402
    Pricing,
    s_per_step_from_csv,
    speed_per_dollar,
    summarise,
)
from gcg_defense.sweep import build_points, to_episodes  # noqa: E402

BENCHMARK_STEPS = 5


def _floats(text: str):
    return [float(v) for v in text.split(",") if v.strip()]


def _ints(text: str):
    return [int(v) for v in text.split(",") if v.strip()]


def cmd_benchmark(args) -> int:
    pricing = Pricing.load(args.pricing)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    print(
        f"Benchmark: {BENCHMARK_STEPS} GCG steps of the DEFAULT config on each GPU.\n"
        "Run these yourself, then pass the resulting steps.csv files to "
        "`compare`.\n"
    )
    for gpu in gpus:
        price = pricing.gpu_price(gpu)
        # a 5-step benchmark: startup + load + 5 steps + one eval
        rough = 30 + 45 + BENCHMARK_STEPS * 20 + 30
        print(f"# {gpu}  (~{rough / 60:.0f} min worst case, ~${price * rough:.2f})")
        print(
            f"GCG_GPU={gpu} modal run --detach scripts/modal_app.py::episode \\\n"
            f"    --config configs/default.yaml \\\n"
            f"    --overrides 'gcg.n_steps={BENCHMARK_STEPS};gcg.eval_every={BENCHMARK_STEPS};"
            f"run.tag=bench_{gpu}' \\\n"
            f"    --gpu {gpu} --skip-sanity\n"
        )
    print(
        "Then fetch the step logs:\n"
        "  modal volume ls gcg-defense-results\n"
        "  modal volume get gcg-defense-results <run_id>/gcg_steps.csv ./\n"
    )
    print(pricing.provenance())
    return 0


def cmd_compare(args) -> int:
    pricing = Pricing.load(args.pricing)
    measured = {}
    for item in args.measured:
        if "=" not in item:
            raise SystemExit(f"--measured expects GPU=path, got {item!r}")
        gpu, path = item.split("=", 1)
        measured[gpu.strip()] = s_per_step_from_csv(path.strip())
    print("Measured seconds per GCG step (median, first step skipped):\n")
    print(speed_per_dollar(measured, pricing))
    best = max(measured, key=lambda g: 1.0 / (pricing.gpu_price(g) * measured[g]))
    print(
        f"\nBest steps per dollar: {best}. Pick the GPU by this column, not by "
        "raw speed -- a card twice as fast at three times the price is a worse buy."
    )
    return 0


def cmd_plan(args) -> int:
    pricing = Pricing.load(args.pricing)
    axis_values = {}
    if args.lambda_util:
        axis_values["lambda_util"] = _floats(args.lambda_util)
    if args.lambda_repr:
        axis_values["lambda_repr"] = _floats(args.lambda_repr)
    if args.suffix_len:
        axis_values["suffix_len"] = _ints(args.suffix_len)
    if args.suffix_position:
        axis_values["suffix_position"] = [
            v.strip() for v in args.suffix_position.split(",") if v.strip()
        ]
    points = build_points(
        config=args.config,
        seeds=_ints(args.seeds),
        gpu=args.gpu,
        axis_values=axis_values,
    )
    episodes = to_episodes(
        points,
        s_per_step={args.gpu: args.s_per_step},
        n_steps=args.n_steps,
        eval_every=args.eval_every,
        eval_seconds=args.eval_seconds,
        evaluation_seconds=args.evaluation_seconds,
    )
    print(summarise(episodes, pricing, max_concurrent=args.max_concurrent))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pricing", default="configs/pricing.json")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="print the benchmark commands")
    bench.add_argument("--gpus", default="A10G,L4,A100,L40S")
    bench.set_defaults(func=cmd_benchmark)

    compare = sub.add_parser("compare", help="speed per dollar from steps.csv files")
    compare.add_argument("--measured", action="append", default=[], metavar="GPU=PATH")
    compare.set_defaults(func=cmd_compare)

    plan = sub.add_parser("plan", help="cost a sweep with a measured s/step")
    plan.add_argument("--config", default="configs/default.yaml")
    plan.add_argument("--gpu", default="A10G")
    plan.add_argument("--s-per-step", type=float, required=True)
    plan.add_argument("--seeds", default="0,1,2")
    plan.add_argument("--lambda-util", default="")
    plan.add_argument("--lambda-repr", default="")
    plan.add_argument("--suffix-len", default="")
    plan.add_argument("--suffix-position", default="")
    plan.add_argument("--n-steps", type=int, default=500)
    plan.add_argument("--eval-every", type=int, default=25)
    plan.add_argument(
        "--eval-seconds",
        type=float,
        default=8.0,
        help="seconds per full-pool evaluation (measured: ~8 s for 256 prompts)",
    )
    plan.add_argument(
        "--evaluation-seconds",
        type=float,
        default=0.0,
        help="seconds for the test-set evaluation stage, if run in the same container",
    )
    plan.add_argument("--max-concurrent", type=int, default=None)
    plan.set_defaults(func=cmd_plan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
