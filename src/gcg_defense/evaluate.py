"""Evaluation of every method on held-out contexts and held-out instructions.

Per method (section 9 of the brief):
  1. model-MI on colluding test contexts (closed form, no sampling);
  2. help rate, mean q(f(a_A));
  3. utility cost: KL on independent test contexts, validity, expected payoff,
     plus the exploitability gap that flags a constant-move policy;
  4. behavioural test: `n_sim_games` simulated games, Bonjour net influence and
     the calibrated within-state permutation test, detection rate and A's
     win-rate advantage over C;
  5. the residual-collusion certificate when nothing is detected;
  6. generalisation: held-out instructions, `info_no_intent`, `private_channel`,
     the intent-only `colluding_implicit` wordings, and transfer models.

Everything lands in `results/<run_id>/metrics.csv` in long format.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from . import detect, io_utils, plots, report
from .baselines import Method, build_methods
from .direction import load_direction_file
from .measure import measure_pool, scalar_metrics
from .model_utils import MoveModel
from .pools import build_pool, build_pools
from .prompts import build_datasets
from .simulate import run_simulation

log = logging.getLogger(__name__)


def _rows(
    method: Method, cfg, model_name: str, split: str, condition: str, metrics: Dict[str, float]
) -> List[Dict[str, object]]:
    return [
        {
            "method": method.name,
            "method_params": method.params,
            "seed": cfg.run.seed,
            "model": model_name,
            "split": split,
            "condition": condition,
            "metric": key,
            "value": value,
        }
        for key, value in metrics.items()
    ]


def evaluate_method_closed_form(
    model: MoveModel,
    cfg,
    method: Method,
    pools: Dict[str, object],
    q_base: torch.Tensor,
    direction: Optional[torch.Tensor] = None,
    repr_layer: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Closed-form metrics for every test pool under one method."""
    out: Dict[str, Dict[str, float]] = {}
    with method.context():
        for name, pool in pools.items():
            metrics = scalar_metrics(
                measure_pool(
                    model,
                    pool,
                    method.suffix_ids,
                    q_base=q_base if name == "independent" else None,
                    repr_layer=repr_layer if direction is not None else None,
                    direction=direction,
                )
            )
            out[name] = metrics
    return out


def evaluate_method_behavioural(
    model: MoveModel, cfg, method: Method, rng_seed: int
) -> Dict[str, Dict[str, float]]:
    """Simulated games, both detectors, and the certificate, per n_rounds.

    Run under two state abstractions: the configured one (by default the
    previous round's joint moves, faithful to Bonjour et al.) and the pooled
    single-state version. Pooling is legitimate here because A and C are
    uniform random and independent of the state, so it cannot manufacture an
    A-B dependence -- and with df = 4 instead of 4*28 it is far more powerful,
    which is what makes the residual certificate informative at these N.
    """
    with method.context():
        sim = run_simulation(
            model,
            cfg,
            suffix_ids=method.suffix_ids,
            condition="colluding",
            seed=rng_seed,
        )
    detect_states = list(dict.fromkeys([cfg.eval.detect_state, "none"]))
    out: Dict[str, Dict[str, float]] = {}
    for detect_state in detect_states:
        n_states = detect.n_states_for(detect_state)
        for n_rounds in cfg.eval.n_rounds:
            prefix = sim.prefix(n_rounds)
            rng = np.random.default_rng(rng_seed + n_rounds)
            detected, bonjour, gammas, nets, p_values = [], [], [], [], []
            for g in range(prefix.actions.shape[0]):
                states = (
                    prefix.states[g]
                    if detect_state == "last_joint_moves"
                    else np.zeros_like(prefix.states[g])
                )
                result = detect.detect_collusion(
                    prefix.actions[g],
                    states,
                    n_states,
                    n_perm=cfg.eval.n_perm,
                    alpha=cfg.eval.alpha,
                    n_pairs=cfg.eval.n_pairs,
                    bonjour_alpha=cfg.eval.bonjour_alpha,
                    rng=rng,
                )
                detected.append(result.detected_ab)
                bonjour.append(result.bonjour_detected_ab)
                gammas.append(result.gamma_ab)
                nets.append(result.net_ab)
                p_values.append(result.p_values[(0, 1)])
            certificate = detect.residual_certificate(
                n_rounds,
                n_states,
                alpha=cfg.eval.alpha,
                n_pairs=cfg.eval.n_pairs,
                power=cfg.eval.power,
            )
            key = f"state={detect_state},n_rounds={n_rounds}"
            out[key] = {
                "detection_rate": float(np.mean(detected)),
                "bonjour_detection_rate": float(np.mean(bonjour)),
                "gamma_ab_mean": float(np.mean(gammas)),
                "net_influence_ab_mean": float(np.mean(nets)),
                "p_value_ab_median": float(np.median(p_values)),
                "a_minus_c_win_rate": prefix.a_advantage_over_c(),
                "a_win_rate": float(prefix.win_rates()[0]),
                "b_win_rate": float(prefix.win_rates()[1]),
                "c_win_rate": float(prefix.win_rates()[2]),
                "help_fraction": prefix.help_fraction(),
                "sim_validity": prefix.mean_validity,
                "certificate_lambda_star": certificate.lambda_star,
                "certificate_df": float(certificate.df),
                "certificate_mi_bound": certificate.mi_bound,
                "certificate_cp_resid": certificate.cp_resid,
            }
            log.info(
                "%s | %-24s N=%4d | detect=%.3f (bonjour %.3f) gamma=%.4f "
                "A-C=%+.4f | %s",
                method.name,
                f"state={detect_state}",
                n_rounds,
                out[key]["detection_rate"],
                out[key]["bonjour_detection_rate"],
                out[key]["gamma_ab_mean"],
                out[key]["a_minus_c_win_rate"],
                certificate.summary(),
            )
    return out


def evaluate_transfer(
    cfg, suffix_text: str, run_dir: str, csv_log
) -> List[Dict[str, object]]:
    """Re-encode the suffix TEXT with each transfer model's own tokeniser."""
    rows: List[Dict[str, object]] = []
    for name in cfg.model.transfer_models:
        try:
            other = MoveModel(
                name,
                device=cfg.run.device,
                dtype=cfg.run.dtype,
                moves=cfg.game.moves,
                batch_size=cfg.model.batch_size,
                use_kv_cache=cfg.model.use_kv_cache,
                max_batch_size=cfg.model.max_batch_size,
            )
        except Exception as exc:  # gated or missing weights
            log.warning("skipping transfer model %s (%s)", name, exc)
            continue
        log.info("transfer: %s | %s", name, other.move_tokens.report())
        datasets = build_datasets(cfg)
        pools = {
            "colluding": build_pool(
                other, "colluding", datasets.test["colluding"], cfg.prompts.suffix_position
            ),
            "independent": build_pool(
                other,
                "independent",
                datasets.test["independent"],
                cfg.prompts.suffix_position,
            ),
        }
        q_base = other.q_batch(pools["independent"].base_prompts, ())["q"].float()
        suffix_ids = other.encode_suffix(suffix_text) if suffix_text else []
        methods = [
            Method(name="none", suffix_ids=[], params="no defence"),
            Method(
                name="gcg_suffix",
                suffix_ids=suffix_ids,
                params=f"transferred text, {len(suffix_ids)} tokens",
            ),
        ]
        for method in methods:
            results = evaluate_method_closed_form(other, cfg, method, pools, q_base)
            for condition, metrics in results.items():
                rows.extend(_rows(method, cfg, name, "test_transfer", condition, metrics))
        del other
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    csv_log.write_all(rows)
    return rows


def run_evaluation(
    cfg,
    run_id: Optional[str] = None,
    suffix_override: Optional[str] = None,
    model: Optional[MoveModel] = None,
) -> Dict[str, object]:
    """Evaluate the GCG suffix and every baseline; write metrics.csv and figures."""
    if run_id:
        cfg.run.run_id = run_id
    run_dir = cfg.run_dir()
    io_utils.setup_logging(run_dir)
    cfg.save(run_dir, "config_eval.yaml")

    suffix_text = suffix_override
    suffix_ids: Optional[List[int]] = None
    best_json = os.path.join(run_dir, "gcg_best_suffix.json")
    if suffix_text is None and os.path.exists(best_json):
        payload = io_utils.read_json(best_json)
        suffix_text = payload["suffix"]
        suffix_ids = list(payload["suffix_ids"])
        log.info("evaluating the suffix from %s: %r", best_json, suffix_text)
    elif suffix_text is not None:
        log.info("evaluating the suffix given on the command line: %r", suffix_text)
    else:
        log.warning("no GCG suffix found in %s; evaluating baselines only", run_dir)

    if model is None:
        model = MoveModel(
            cfg.model.name,
            device=cfg.run.device,
            dtype=cfg.run.dtype,
            moves=cfg.game.moves,
            batch_size=cfg.model.batch_size,
            use_kv_cache=cfg.model.use_kv_cache,
            max_batch_size=cfg.model.max_batch_size,
        )
    log.info(model.move_tokens.report())
    if suffix_ids is None and suffix_text:
        suffix_ids = model.encode_suffix(suffix_text)

    direction = None
    repr_layer = cfg.loss.repr_layer
    std_independent = 1.0
    gap_in_std = float("nan")
    direction_path = cfg.loss.collusion_direction_path or os.path.join(
        cfg.run.out_dir, "collusion_direction.pt"
    )
    if os.path.exists(direction_path):
        direction, repr_layer, _, std_independent, gap_in_std = load_direction_file(
            direction_path, device=model.device
        )
        log.info(
            "using the collusion direction from %s (layer %d, std %.4f; colluding "
            "prompts sit %.1f std from independent ones, so steering by 2-8 std is "
            "a fraction of that gap)",
            direction_path,
            repr_layer,
            std_independent,
            gap_in_std,
        )

    datasets = build_datasets(cfg)
    pools = build_pools(model, datasets, cfg, splits=("test",))["test"]
    q_base = model.q_batch(pools["independent"].base_prompts, ())["q"].float()

    methods = build_methods(
        model,
        cfg,
        gcg_suffix_ids=suffix_ids,
        direction=direction,
        repr_layer=repr_layer,
        std_independent=std_independent,
        gap_in_std=gap_in_std,
    )
    log.info("evaluating %d methods: %s", len(methods), [m.name for m in methods])

    csv_log = io_utils.metrics_log(run_dir, "metrics.csv")
    summary: Dict[str, Dict[str, float]] = {}
    for method in methods:
        closed = evaluate_method_closed_form(
            model, cfg, method, pools, q_base, direction, repr_layer
        )
        for condition, metrics in closed.items():
            csv_log.write_all(_rows(method, cfg, cfg.model.name, "test", condition, metrics))
        flags = flag_degenerate(cfg, closed)
        csv_log.write_all(_rows(method, cfg, cfg.model.name, "test", "flags", flags))
        behavioural = evaluate_method_behavioural(model, cfg, method, cfg.run.seed)
        for key, metrics in behavioural.items():
            csv_log.write_all(
                _rows(method, cfg, cfg.model.name, "test_sim", key, metrics)
            )
        summary[method.name] = {
            "model_mi": closed["colluding"]["model_mi"],
            "model_mi_sem": closed["colluding"]["model_mi_sem"],
            "help_rate": closed["colluding"]["help_rate"],
            "util_kl": closed["independent"]["util_kl"],
            "util_kl_sem": closed["independent"]["util_kl_sem"],
            "validity": closed["independent"]["validity"],
            "entropy": closed["independent"]["entropy"],
            "best_response_gap": closed["independent"]["best_response_gap"],
            "mi_info_no_intent": closed["info_no_intent"]["model_mi"],
            "mi_private_channel": closed["private_channel"]["model_mi"],
            "mi_colluding_implicit": closed["colluding_implicit"]["model_mi"],
            "mi_colluding_train_instr": closed["colluding_train_instr"]["model_mi"],
            "constant_move_flag": flags["constant_move_flag"],
            **{
                f"{metric}_{_rounds_of(k)}": v[metric]
                for k, v in behavioural.items()
                if _state_of(k) == cfg.eval.detect_state
                for metric in ("detection_rate", "a_minus_c_win_rate", "certificate_cp_resid")
            },
            **{
                f"detection_rate_pooled_{_rounds_of(k)}": v["detection_rate"]
                for k, v in behavioural.items()
                if _state_of(k) == "none"
            },
            **{
                f"cp_resid_pooled_{_rounds_of(k)}": v["certificate_cp_resid"]
                for k, v in behavioural.items()
                if _state_of(k) == "none"
            },
        }
        log.info(
            "%-22s MI=%.5f help=%.3f util_kl=%.5f validity=%.4f",
            method.name,
            summary[method.name]["model_mi"],
            summary[method.name]["help_rate"],
            summary[method.name]["util_kl"],
            summary[method.name]["validity"],
        )

    if cfg.eval.transfer and suffix_text:
        evaluate_transfer(cfg, suffix_text, run_dir, csv_log)
    csv_log.close()

    io_utils.write_json(os.path.join(run_dir, "summary.json"), summary)
    io_utils.write_text(
        os.path.join(run_dir, "summary.txt"), format_summary(summary, cfg) + "\n"
    )
    log.info("summary\n%s", format_summary(summary, cfg))
    figures = plots.make_all(run_dir, cfg)
    report_path = report.write_report(run_dir)
    log.info("wrote %s", report_path)
    return {
        "run_dir": run_dir,
        "summary": summary,
        "figures": figures,
        "report": report_path,
    }


def _state_of(key: str) -> str:
    return key.split(",")[0].split("=")[1]


def _rounds_of(key: str) -> str:
    return key.split("n_rounds=")[1]


def flag_degenerate(cfg, closed: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """A constant-move B has zero MI but is exploitable: flag it."""
    independent = closed["independent"]
    constant = float(independent["max_prob"] >= cfg.sanity.constant_move_threshold)
    return {
        "constant_move_flag": constant,
        "max_prob_independent": independent["max_prob"],
        "best_response_gap": independent["best_response_gap"],
        "entropy_independent": independent["entropy"],
        "validity_flag": float(independent["validity"] < cfg.sanity.min_validity),
    }


def format_summary(summary: Dict[str, Dict[str, float]], cfg) -> str:
    header = (
        f"{'method':<22}{'MI':>9}{'help':>7}{'utilKL':>9}{'valid':>7}"
        f"{'H(q)':>7}{'expl':>7}"
        f"{'det@' + str(max(cfg.eval.n_rounds)):>9}{'detPool':>9}{'flag':>6}"
    )
    lines = [header, "-" * len(header)]
    for name, m in summary.items():
        rounds = max(cfg.eval.n_rounds)
        detection = m.get(f"detection_rate_{rounds}", float("nan"))
        pooled = m.get(f"detection_rate_pooled_{rounds}", float("nan"))
        lines.append(
            f"{name:<22}{m['model_mi']:>9.5f}{m['help_rate']:>7.3f}"
            f"{m['util_kl']:>9.5f}{m['validity']:>7.4f}{m['entropy']:>7.3f}"
            f"{m['best_response_gap']:>7.3f}{detection:>9.3f}{pooled:>9.3f}"
            f"{int(m['constant_move_flag']):>6d}"
        )
    return "\n".join(lines)
