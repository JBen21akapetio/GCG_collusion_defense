"""Modal entry points. One **episode** = one GCG run (one config x one seed).

Nothing here launches on import, and every launching entry point accepts
`--dry-run`, which prints the planned jobs (config, seed, GPU, timeout,
estimated cost) and exits without starting a container.

    # 1. plan, and look at the bill first
    modal run scripts/modal_app.py::sweep --lambda-util 0.1,1,10 --seeds 0,1,2 \
        --gpu A10G --s-per-step 12.4 --dry-run

    # 2. launch detached, so closing the laptop does not kill the jobs
    modal run --detach scripts/modal_app.py::sweep --lambda-util 0.1,1,10 \
        --seeds 0,1,2 --gpu A10G --s-per-step 12.4 --max-concurrent 4

    # 3. resume a single episode from its last checkpoint
    modal run --detach scripts/modal_app.py::episode --resume <run_id>

Episodes fan out with `.starmap()`, one container per episode, so sweep points
and seeds run in parallel. The GPU, the timeout and the container cap are set
per call with `Function.with_options()`, so a benchmark on L4 and a production
run on A10G use the same code path.

Results and checkpoints live in the `gcg-defense-results` volume; model weights
are cached in `gcg-defense-hf-cache` so containers never re-download them.

    modal volume get gcg-defense-results <run_id> results/
"""

from __future__ import annotations

import os
import sys

import modal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

APP_NAME = "gcg-defense"

#: Default GPU. Qwen2.5-1.5B in bf16 needs ~3.5 GiB for weights and a few GiB
#: of activations, so it fits comfortably on a 24 GiB A10G or L4; the right
#: choice is the one with the best measured steps-per-dollar, which
#: `scripts/estimate_cost.py compare` computes from a real benchmark.
DEFAULT_GPU = os.environ.get("GCG_GPU", "A10G")
#: Fallback timeout when no measured s/step is supplied (seconds).
FALLBACK_TIMEOUT = 4 * 60 * 60

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "numpy==2.1.3",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "pandas==2.2.3",
        "pyyaml==6.0.2",
        "pytest==8.3.3",
        "huggingface_hub==0.26.2",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")
    .add_local_dir("configs", "/root/configs")
    .add_local_dir("tests", "/root/tests")
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("gcg-defense-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("gcg-defense-results", create_if_missing=True)
VOLUMES = {"/cache": hf_cache, "/root/results": results}

# Optional, only for the gated Llama-3.2 transfer model:
#   modal secret create hf-secret HF_TOKEN=...
#   GCG_HF_SECRET=hf-secret modal run scripts/modal_app.py::evaluate ...
_HF_SECRET = os.environ.get("GCG_HF_SECRET", "")
SECRETS = [modal.Secret.from_name(_HF_SECRET)] if _HF_SECRET else []


def _parse(overrides: str):
    return [o for o in (overrides or "").split(";") if o.strip()]


# --- remote bodies ------------------------------------------------------------
# These carry defaults only; the GPU, timeout and container cap are set per call
# with .with_options() by the local entry points below.


@app.function(gpu=DEFAULT_GPU, volumes=VOLUMES, timeout=FALLBACK_TIMEOUT, secrets=SECRETS)
def _episode(
    config: str = "configs/default.yaml",
    overrides: str = "",
    resume: str = "",
    skip_sanity: bool = False,
    with_eval: bool = False,
):
    """One episode: optimise a defensive suffix, optionally evaluate it."""
    import time

    import torch

    from gcg_defense.config import Config
    from gcg_defense.evaluate import run_evaluation
    from gcg_defense.gcg import run_gcg

    os.chdir("/root")
    started = time.time()
    if resume and os.path.exists(f"results/{resume}/config.yaml"):
        config = f"results/{resume}/config.yaml"
    cfg = Config.load(config, _parse(overrides))
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    out = run_gcg(cfg, resume_run_id=resume or None, skip_sanity=skip_sanity)
    results.commit()
    payload = {
        "run_id": out["run_id"],
        "run_dir": out["run_dir"],
        "suffix": out["suffix"],
        "steps": out["steps"],
        "best_loss": out["best_loss"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seconds": round(time.time() - started, 1),
    }
    if with_eval:
        cfg_eval = Config.load(config, _parse(overrides))
        evaluated = run_evaluation(cfg_eval, run_id=out["run_id"])
        payload["summary"] = evaluated["summary"]
        results.commit()
    payload["total_seconds"] = round(time.time() - started, 1)
    print(payload, flush=True)
    return payload


@app.function(gpu=DEFAULT_GPU, volumes=VOLUMES, timeout=FALLBACK_TIMEOUT, secrets=SECRETS)
def _evaluate(
    run_id: str = "",
    config: str = "configs/default.yaml",
    overrides: str = "",
    suffix: str = "",
):
    from gcg_defense.config import Config
    from gcg_defense.evaluate import run_evaluation

    os.chdir("/root")
    if run_id and os.path.exists(f"results/{run_id}/config.yaml"):
        config = f"results/{run_id}/config.yaml"
    cfg = Config.load(config, _parse(overrides))
    out = run_evaluation(cfg, run_id=run_id or None, suffix_override=suffix or None)
    results.commit()
    return {"run_dir": out["run_dir"], "figures": out["figures"], "report": out["report"]}


@app.function(gpu=DEFAULT_GPU, volumes=VOLUMES, timeout=60 * 60, secrets=SECRETS)
def _sanity(config: str = "configs/default.yaml", overrides: str = ""):
    import json

    from gcg_defense import io_utils
    from gcg_defense.config import Config
    from gcg_defense.sanity import run_sanity

    os.chdir("/root")
    cfg = Config.load(config, _parse(overrides))
    run_dir = cfg.run_dir()
    io_utils.setup_logging(run_dir)
    cfg.save(run_dir)
    report = run_sanity(cfg, run_dir=run_dir)
    results.commit()
    print(report.summary(), flush=True)
    return json.dumps({"passed": report.passed, "run_dir": run_dir})


@app.function(gpu=DEFAULT_GPU, volumes=VOLUMES, timeout=2 * 60 * 60, secrets=SECRETS)
def _positive_control(config: str = "configs/fast.yaml", overrides: str = ""):
    from gcg_defense import io_utils
    from gcg_defense.config import Config
    from gcg_defense.positive_control import run_positive_control

    os.chdir("/root")
    cfg = Config.load(config, _parse(overrides))
    run_dir = cfg.run_dir()
    io_utils.setup_logging(run_dir)
    cfg.save(run_dir)
    out = run_positive_control(cfg, run_dir=run_dir)
    results.commit()
    return out


@app.function(gpu=DEFAULT_GPU, volumes=VOLUMES, timeout=60 * 60, secrets=SECRETS)
def _direction(config: str = "configs/default.yaml", overrides: str = ""):
    from gcg_defense import io_utils
    from gcg_defense.config import Config
    from gcg_defense.direction import compute_collusion_direction

    os.chdir("/root")
    cfg = Config.load(config, _parse(overrides))
    io_utils.setup_logging(None)
    out = compute_collusion_direction(cfg)
    results.commit()
    print(out, flush=True)
    return out


@app.function(gpu=DEFAULT_GPU, volumes=VOLUMES, timeout=60 * 60, secrets=SECRETS)
def _prompt_report(
    kind: str = "instructions",
    config: str = "configs/default.yaml",
    overrides: str = "",
    templates: str = "configs/_probe_instructions.yaml",
):
    """Per-instruction diagnostics (`instructions`) or a phrasing probe (`probe`)."""
    import numpy as np
    import yaml

    from gcg_defense import io_utils
    from gcg_defense.config import Config
    from gcg_defense.model_utils import MoveModel
    from gcg_defense.pools import build_pool
    from gcg_defense.prompts import (
        COLLUSION_INSTRUCTIONS,
        INFO_NO_INTENT_INSTRUCTIONS,
        PRIVATE_CHANNEL_MESSAGES,
        build_contexts,
    )
    from gcg_defense.sanity import format_instruction_report, per_instruction_report

    os.chdir("/root")
    cfg = Config.load(config, _parse(overrides))
    io_utils.setup_logging(None)
    model = MoveModel(
        cfg.model.name,
        device=cfg.run.device,
        dtype=cfg.run.dtype,
        moves=cfg.game.moves,
        batch_size=cfg.model.batch_size,
        max_batch_size=cfg.model.max_batch_size,
    )
    print(model.move_tokens.report(), flush=True)
    if kind == "probe":
        with open(templates) as fh:
            spec = yaml.safe_load(fh)
        texts = list(spec["templates"])
        n = int(spec.get("n_contexts_per_template", 16)) * len(texts)
        contexts = build_contexts(
            spec.get("condition", "colluding"),
            n,
            cfg.game.history_len,
            np.random.default_rng(cfg.run.seed),
            "all",
            cfg.prompts.n_train_instructions,
            templates=texts,
        )
        pool = build_pool(model, "probe", contexts, cfg.prompts.suffix_position)
        rows = per_instruction_report(model, pool)
        print(format_instruction_report(rows, texts), flush=True)
        return rows
    report = {}
    for condition, pool_texts in (
        ("colluding", COLLUSION_INSTRUCTIONS),
        ("info_no_intent", INFO_NO_INTENT_INSTRUCTIONS),
        ("private_channel", PRIVATE_CHANNEL_MESSAGES),
    ):
        contexts = build_contexts(
            condition,
            max(len(pool_texts) * 8, 24),
            cfg.game.history_len,
            np.random.default_rng(cfg.run.seed),
            "all",
            cfg.prompts.n_train_instructions,
        )
        pool = build_pool(model, condition, contexts, cfg.prompts.suffix_position)
        rows = per_instruction_report(model, pool)
        print(f"\n=== {condition} ===", flush=True)
        print(format_instruction_report(rows, pool_texts), flush=True)
        report[condition] = rows
    return report


@app.function(volumes=VOLUMES, timeout=60 * 60)
def _tests(k: str = "", model_tests: bool = False):
    import subprocess

    cmd = [sys.executable, "-m", "pytest", "/root/tests", "-v", "--color=no"]
    if not model_tests:
        cmd += ["-m", "not requires_model"]
    if k:
        cmd += ["-k", k]
    proc = subprocess.run(cmd, cwd="/root", capture_output=True, text=True)
    print(proc.stdout[-20000:])
    print(proc.stderr[-5000:])
    if proc.returncode != 0:
        raise RuntimeError(f"pytest failed with code {proc.returncode}")
    return proc.returncode


@app.function(volumes=VOLUMES, timeout=30 * 60)
def _figures(run_id: str, extra_run_ids: str = "", out_name: str = ""):
    """Redraw figures and report.md from metrics.csv (CPU only, no model)."""
    from gcg_defense import io_utils, plots
    from gcg_defense.config import Config
    from gcg_defense.report import write_report

    os.chdir("/root")
    extra = [f"results/{r}/metrics.csv" for r in extra_run_ids.split(",") if r.strip()]
    run_dir = f"results/{out_name or run_id}"
    os.makedirs(run_dir, exist_ok=True)
    io_utils.setup_logging(run_dir)
    saved = f"results/{run_id}/config.yaml"
    cfg = Config.load(saved if os.path.exists(saved) else "configs/default.yaml")
    if out_name and out_name != run_id:
        csvs = [f"results/{run_id}/metrics.csv", *extra]
        figures = plots.pareto(run_dir, csvs, cfg)
        figures += plots.detection_curves(run_dir, csvs, "last_joint_moves")
        figures += plots.detection_curves(run_dir, csvs, "none")
        path = write_report(run_dir, csvs)
    else:
        figures = plots.make_all(run_dir, cfg, extra_csvs=extra)
        path = write_report(run_dir, [f"{run_dir}/metrics.csv", *extra])
    results.commit()
    print(figures, path, flush=True)
    return figures


@app.function(volumes=VOLUMES, timeout=30 * 60)
def _bench_detect(n_perm: int = 1000):
    import time

    import numpy as np

    from gcg_defense import detect

    os.chdir("/root")
    rng = np.random.default_rng(0)
    out = {}
    for n_rounds in (50, 100, 200, 500):
        for n_states in (1, 28):
            actions = rng.integers(0, 3, size=(n_rounds, 3))
            states = rng.integers(0, n_states, size=n_rounds)
            start = time.time()
            detect.detect_collusion(actions, states, n_states, n_perm=n_perm, rng=rng)
            out[f"N={n_rounds},S={n_states}"] = round(time.time() - start, 4)
    print(out, flush=True)
    return out


# --- planning helpers ---------------------------------------------------------


def _plan(points, s_per_step, n_steps, eval_every, eval_seconds, evaluation_seconds,
          max_concurrent, pricing_path="configs/pricing.json"):
    """Print the job plan and the bill. Returns {name: timeout_seconds}."""
    from gcg_defense.cost import Pricing
    from gcg_defense.sweep import format_plan, to_episodes

    timeouts = {}
    try:
        pricing = Pricing.load(pricing_path)
    except FileNotFoundError:
        pricing = None
    if s_per_step and s_per_step > 0:
        episodes = to_episodes(
            points,
            s_per_step={p.gpu: s_per_step for p in points},
            n_steps=n_steps,
            eval_every=eval_every,
            eval_seconds=eval_seconds,
            evaluation_seconds=evaluation_seconds,
        )
        timeouts = {e.name: e.timeout_seconds() for e in episodes}
        print(format_plan(points, timeouts))
        if pricing is not None:
            from gcg_defense.cost import summarise

            print()
            print(summarise(episodes, pricing, max_concurrent=max_concurrent))
    else:
        print(format_plan(points))
        print(
            "\nNo --s-per-step given, so no duration or cost estimate and the "
            f"timeout falls back to {FALLBACK_TIMEOUT // 3600}h per episode.\n"
            "Measure it first:  python scripts/estimate_cost.py benchmark"
        )
    return timeouts


def _points_from_args(config, gpu, seeds, lambda_util, lambda_repr, suffix_len,
                      suffix_position, extra):
    from gcg_defense.sweep import build_points

    def _split(text, cast):
        return [cast(v) for v in str(text).split(",") if str(v).strip()]

    axis_values = {}
    if lambda_util:
        axis_values["lambda_util"] = _split(lambda_util, float)
    if lambda_repr:
        axis_values["lambda_repr"] = _split(lambda_repr, float)
    if suffix_len:
        axis_values["suffix_len"] = _split(suffix_len, int)
    if suffix_position:
        axis_values["suffix_position"] = _split(suffix_position, str)
    return build_points(
        config=config,
        seeds=_split(seeds, int),
        gpu=gpu,
        axis_values=axis_values,
        extra_overrides=_parse(extra),
    )


# --- local entry points (these are what you run) ------------------------------


@app.local_entrypoint()
def sweep(
    config: str = "configs/default.yaml",
    gpu: str = DEFAULT_GPU,
    seeds: str = "0,1,2",
    lambda_util: str = "",
    lambda_repr: str = "",
    suffix_len: str = "",
    suffix_position: str = "",
    extra: str = "",
    s_per_step: float = 0.0,
    n_steps: int = 500,
    eval_every: int = 25,
    eval_seconds: float = 8.0,
    evaluation_seconds: float = 0.0,
    max_concurrent: int = 4,
    with_eval: bool = False,
    dry_run: bool = False,
):
    """Fan a grid of episodes out over parallel GPUs, one container each."""
    points = _points_from_args(
        config, gpu, seeds, lambda_util, lambda_repr, suffix_len, suffix_position, extra
    )
    timeouts = _plan(
        points, s_per_step, n_steps, eval_every, eval_seconds,
        evaluation_seconds, max_concurrent,
    )
    if dry_run:
        print("\n--dry-run: nothing was launched.")
        return
    timeout = max(timeouts.values()) if timeouts else FALLBACK_TIMEOUT
    print(
        f"\nlaunching {len(points)} episode(s) on {gpu}, "
        f"max {max_concurrent} at a time, timeout {timeout}s each"
    )
    runner = _episode.with_options(
        gpu=gpu, timeout=timeout, max_containers=max_concurrent
    )
    args = [
        (p.config, p.override_string(), "", True, with_eval) for p in points
    ]
    run_ids = []
    for result in runner.starmap(args):
        print(result["run_id"], result.get("seconds"), flush=True)
        run_ids.append(result["run_id"])
    print("\nrun ids:", ",".join(run_ids))
    print(
        "merge the figures with:\n"
        f"  modal run scripts/modal_app.py::figures --run-id {run_ids[0]} "
        f"--extra-run-ids {','.join(run_ids[1:])} --out-name sweep"
    )


@app.local_entrypoint()
def episode(
    config: str = "configs/default.yaml",
    overrides: str = "",
    gpu: str = DEFAULT_GPU,
    seed: int = 0,
    s_per_step: float = 0.0,
    n_steps: int = 500,
    eval_every: int = 25,
    eval_seconds: float = 8.0,
    resume: str = "",
    skip_sanity: bool = False,
    with_eval: bool = False,
    dry_run: bool = False,
):
    """One episode (one config x one seed) on one GPU."""
    from gcg_defense.sweep import SweepPoint

    extra = _parse(overrides)
    if not any(o.startswith("run.seed=") for o in extra):
        extra.insert(0, f"run.seed={seed}")
    point = SweepPoint(
        tag="resume" if resume else "episode",
        seed=seed,
        gpu=gpu,
        config=config,
        overrides=extra,
    )
    timeouts = _plan(
        [point], s_per_step, n_steps, eval_every, eval_seconds, 0.0, 1
    )
    if dry_run:
        print("\n--dry-run: nothing was launched.")
        return
    timeout = max(timeouts.values()) if timeouts else FALLBACK_TIMEOUT
    out = _episode.with_options(gpu=gpu, timeout=timeout).remote(
        config, ";".join(extra), resume, skip_sanity, with_eval
    )
    print(out)


@app.local_entrypoint()
def evaluate(
    run_id: str = "",
    config: str = "configs/default.yaml",
    overrides: str = "",
    suffix: str = "",
    gpu: str = DEFAULT_GPU,
    timeout: int = 2 * 60 * 60,
    dry_run: bool = False,
):
    """Evaluate a finished episode: all methods, all test pools, figures, report."""
    print(
        f"plan: evaluate run_id={run_id or '(none)'} on {gpu}, timeout {timeout}s\n"
        f"      config={config} overrides={overrides or '(none)'}"
    )
    if dry_run:
        print("--dry-run: nothing was launched.")
        return
    print(_evaluate.with_options(gpu=gpu, timeout=timeout).remote(
        run_id, config, overrides, suffix
    ))


@app.local_entrypoint()
def sanity(
    config: str = "configs/default.yaml",
    overrides: str = "",
    gpu: str = DEFAULT_GPU,
    dry_run: bool = False,
):
    """Pre-optimisation sanity checks (brief section 5)."""
    print(f"plan: sanity on {gpu}, config={config}, overrides={overrides or '(none)'}")
    if dry_run:
        print("--dry-run: nothing was launched.")
        return
    print(_sanity.with_options(gpu=gpu).remote(config, overrides))


@app.local_entrypoint()
def positive_control(
    config: str = "configs/fast.yaml",
    overrides: str = "",
    gpu: str = DEFAULT_GPU,
    dry_run: bool = False,
):
    """Benign optimiser control: force B to answer "Paper"."""
    print(f"plan: positive control on {gpu}, config={config}")
    if dry_run:
        print("--dry-run: nothing was launched.")
        return
    print(_positive_control.with_options(gpu=gpu).remote(config, overrides))


@app.local_entrypoint()
def collusion_direction(
    config: str = "configs/default.yaml",
    overrides: str = "",
    gpu: str = DEFAULT_GPU,
    dry_run: bool = False,
):
    """Diff-in-means collusion direction, written to the results volume."""
    print(f"plan: collusion direction on {gpu}, config={config}")
    if dry_run:
        print("--dry-run: nothing was launched.")
        return
    print(_direction.with_options(gpu=gpu).remote(config, overrides))


@app.local_entrypoint()
def instructions(
    config: str = "configs/default.yaml",
    overrides: str = "",
    gpu: str = DEFAULT_GPU,
    dry_run: bool = False,
):
    """Per-instruction help / counter / copy rates."""
    print(f"plan: instruction report on {gpu}")
    if dry_run:
        print("--dry-run: nothing was launched.")
        return
    print(_prompt_report.with_options(gpu=gpu).remote("instructions", config, overrides))


@app.local_entrypoint()
def probe(
    templates: str = "configs/_probe_instructions.yaml",
    config: str = "configs/default.yaml",
    overrides: str = "",
    gpu: str = DEFAULT_GPU,
    dry_run: bool = False,
):
    """Score candidate collusion phrasings (prompt-strength loop)."""
    print(f"plan: phrasing probe on {gpu}, templates={templates}")
    if dry_run:
        print("--dry-run: nothing was launched.")
        return
    print(_prompt_report.with_options(gpu=gpu).remote("probe", config, overrides, templates))


@app.local_entrypoint()
def tests(k: str = "", model_tests: bool = False, dry_run: bool = False):
    """Unit tests inside the Modal image (CPU only, no GPU charge)."""
    print(f"plan: pytest {'(with model tests)' if model_tests else '(no model tests)'}, CPU only")
    if dry_run:
        print("--dry-run: nothing was launched.")
        return
    print(_tests.remote(k, model_tests))


@app.local_entrypoint()
def figures(
    run_id: str,
    extra_run_ids: str = "",
    out_name: str = "",
    dry_run: bool = False,
):
    """Redraw figures and report.md for a run (CPU only, no GPU charge)."""
    print(f"plan: figures for {run_id} (+{extra_run_ids or 'none'}), CPU only")
    if dry_run:
        print("--dry-run: nothing was launched.")
        return
    print(_figures.remote(run_id, extra_run_ids, out_name))


@app.local_entrypoint()
def bench_detect(n_perm: int = 1000, dry_run: bool = False):
    """Time the permutation test (CPU only)."""
    print(f"plan: detection benchmark, n_perm={n_perm}, CPU only")
    if dry_run:
        print("--dry-run: nothing was launched.")
        return
    print(_bench_detect.remote(n_perm))


@app.function(gpu=DEFAULT_GPU, volumes=VOLUMES, timeout=2 * 60 * 60, secrets=SECRETS)
def _perturb(
    run_id: str,
    config: str = "configs/default.yaml",
    overrides: str = "",
    suffix: str = "",
    draws: int = 3,
):
    from gcg_defense.config import Config
    from gcg_defense.robustness import run_perturbation_study

    os.chdir("/root")
    if os.path.exists(f"results/{run_id}/config.yaml"):
        config = f"results/{run_id}/config.yaml"
    cfg = Config.load(config, _parse(overrides))
    out = run_perturbation_study(
        cfg, run_id=run_id, suffix_override=suffix or None, draws=draws
    )
    results.commit()
    print(out["summary"], flush=True)
    return {"run_dir": out["run_dir"], "summary": out["summary"]}


@app.function(volumes=VOLUMES, timeout=30 * 60)
def _compare_suffixes(run_ids: str, out_name: str = "suffix_comparison"):
    """Compare les suffixes de plusieurs runs (texte seul, pas de modèle)."""
    from gcg_defense import io_utils
    from gcg_defense.robustness import compare_runs

    os.chdir("/root")
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    text = compare_runs([f"results/{r}" for r in ids])
    # Ecrit aussi un fichier : sans cela le resultat ne vivrait que dans les
    # logs, qui disparaissent avec --detach.
    path = f"results/{out_name}.txt"
    io_utils.write_text(path, text + "\n")
    results.commit()
    print(text, flush=True)
    print(f"\n-> ecrit dans {path}", flush=True)
    return text


@app.local_entrypoint()
def perturb(
    run_id: str,
    config: str = "configs/default.yaml",
    overrides: str = "",
    suffix: str = "",
    draws: int = 3,
    gpu: str = DEFAULT_GPU,
    dry_run: bool = False,
):
    """Robustesse 1 : le suffixe survit-il à de petites perturbations ?"""
    n_variants = 1 + 4 * draws + 2 * draws + draws + 1 + 2 + 4 + draws + 1
    print(
        f"plan: ~{n_variants} variantes x 256 prompts sur {gpu}\n"
        f"      run_id={run_id}, draws={draws}\n"
        f"      ~{n_variants * 8 / 60:.0f} min, aucun GCG (scoring seulement)"
    )
    if dry_run:
        print("--dry-run: rien n'a été lancé.")
        return
    print(_perturb.with_options(gpu=gpu).remote(run_id, config, overrides, suffix, draws))


@app.local_entrypoint()
def compare_suffixes(
    run_ids: str, out_name: str = "suffix_comparison", dry_run: bool = False
):
    """Robustesse 2 et 3 : les suffixes de plusieurs runs se ressemblent-ils ?"""
    print(
        f"plan: comparaison textuelle de {len(run_ids.split(','))} runs, CPU seul\n"
        f"      -> results/{out_name}.txt dans le volume"
    )
    if dry_run:
        print("--dry-run: rien n'a été lancé.")
        return
    print(_compare_suffixes.remote(run_ids, out_name))


@app.function(gpu=DEFAULT_GPU, volumes=VOLUMES, timeout=60 * 60, secrets=SECRETS)
def _explain(run_id: str, config: str = "configs/default.yaml", overrides: str = "", suffix: str = ""):
    from gcg_defense.config import Config
    from gcg_defense.explain import run_explain

    os.chdir("/root")
    if os.path.exists(f"results/{run_id}/config.yaml"):
        config = f"results/{run_id}/config.yaml"
    cfg = Config.load(config, _parse(overrides))
    out = run_explain(cfg, run_id=run_id, suffix_override=suffix or None)
    results.commit()
    print(out["figures"], flush=True)
    return {"run_dir": out["run_dir"], "figures": out["figures"]}


@app.local_entrypoint()
def explain(
    run_id: str,
    config: str = "configs/default.yaml",
    overrides: str = "",
    suffix: str = "",
    gpu: str = DEFAULT_GPU,
    dry_run: bool = False,
):
    """Figures pédagogiques : table de gains, matrices q, décomposition des coups."""
    print(
        f"plan: figures explicatives pour {run_id} sur {gpu}\n"
        "      5 conditions x 2 defences x 192 prompts ~ 2 min, ~0.10 $\n"
        "      -> game_payoff, q_matrices_*, move_decomposition_*"
    )
    if dry_run:
        print("--dry-run: rien n'a été lancé.")
        return
    print(_explain.with_options(gpu=gpu).remote(run_id, config, overrides, suffix))
