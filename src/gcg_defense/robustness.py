"""Robustness of a found suffix: perturbations, and comparison across runs.

Three questions, all about whether the suffix is a knife-edge point in token
space or a region:

1. **Perturbation** -- does a suffix that differs by a few characters still
   remove the collusion? `run_perturbation_study` scores controlled variants
   (replace k tokens, delete, shuffle the order, change case, strip spaces,
   truncate) against the untouched suffix and a random control.
2. **Across models** -- does GCG converge to the same string on a different
   model? Run an episode per model, then `compare_runs` on the suffixes.
   Only text-level comparison is meaningful across tokenisers.
3. **Across attackers** -- does training on a different set of collusion
   instructions produce a different suffix? Same comparison, after an episode
   with `prompts.train_instruction_ids` set to another split.

The `shuffle` variant is the informative one for (1): it keeps exactly the same
tokens and only changes their order. If the defence survives it, the suffix
acts as a bag of tokens; if it collapses, position carries the effect.
"""

from __future__ import annotations

import difflib
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import io_utils
from .measure import measure_pool, scalar_metrics
from .model_utils import MoveModel
from .pools import build_pools
from .prompts import build_datasets

log = logging.getLogger(__name__)


@dataclass
class Variant:
    """One perturbed suffix to score."""

    name: str
    kind: str
    suffix_ids: List[int]
    draw: int = 0
    magnitude: float = 0.0

    @property
    def n_tokens(self) -> int:
        return len(self.suffix_ids)


# --- building the variants ----------------------------------------------------

def _allowed_ids(model: MoveModel, allow_non_ascii: bool = False) -> np.ndarray:
    return model.allowed_token_mask(allow_non_ascii).nonzero().flatten().cpu().numpy()


def _retokenise(model: MoveModel, text: str) -> List[int]:
    return model.encode_suffix(text)


def make_variants(
    model: MoveModel,
    suffix_ids: Sequence[int],
    rng: np.random.Generator,
    draws: int = 3,
    replace_counts: Sequence[int] = (1, 2, 5, 10),
    delete_counts: Sequence[int] = (1, 5),
    allow_non_ascii: bool = False,
) -> List[Variant]:
    """The perturbation ladder, from "one token changed" to "unrelated string"."""
    base = list(suffix_ids)
    k = len(base)
    allowed = _allowed_ids(model, allow_non_ascii)
    variants: List[Variant] = [
        Variant(name="original", kind="original", suffix_ids=list(base))
    ]

    for count in replace_counts:
        if count > k:
            continue
        for draw in range(draws):
            ids = list(base)
            positions = rng.choice(k, size=count, replace=False)
            for pos in positions:
                ids[int(pos)] = int(rng.choice(allowed))
            variants.append(
                Variant(
                    name=f"replace_{count}_d{draw}",
                    kind=f"replace_{count}",
                    suffix_ids=ids,
                    draw=draw,
                    magnitude=count / k,
                )
            )

    for count in delete_counts:
        if count >= k:
            continue
        for draw in range(draws):
            keep = sorted(rng.choice(k, size=k - count, replace=False))
            variants.append(
                Variant(
                    name=f"delete_{count}_d{draw}",
                    kind=f"delete_{count}",
                    suffix_ids=[base[int(i)] for i in keep],
                    draw=draw,
                    magnitude=count / k,
                )
            )

    # same tokens, different order: separates "bag of tokens" from "sequence"
    for draw in range(draws):
        ids = list(base)
        rng.shuffle(ids)
        variants.append(
            Variant(name=f"shuffle_d{draw}", kind="shuffle", suffix_ids=ids, draw=draw)
        )
    variants.append(
        Variant(name="reverse", kind="reverse", suffix_ids=list(reversed(base)))
    )

    # halves: is the effect carried by one part?
    variants.append(
        Variant(name="first_half", kind="truncate", suffix_ids=base[: k // 2])
    )
    variants.append(
        Variant(name="second_half", kind="truncate", suffix_ids=base[k // 2 :])
    )

    # character-level edits, re-encoded (the token ids change a lot)
    text = model.decode_suffix(base)
    for name, edited in (
        ("lowercase", text.lower()),
        ("uppercase", text.upper()),
        ("strip_spaces", " ".join(text.split())),
        ("no_newlines", text.replace("\n", " ")),
    ):
        ids = _retokenise(model, edited)
        if ids and ids != base:
            variants.append(
                Variant(name=name, kind="char_edit", suffix_ids=ids)
            )

    # controls
    for draw in range(draws):
        ids = [int(t) for t in rng.choice(allowed, size=k, replace=True)]
        variants.append(
            Variant(name=f"random_d{draw}", kind="random", suffix_ids=ids, draw=draw, magnitude=1.0)
        )
    variants.append(Variant(name="empty", kind="none", suffix_ids=[]))
    return variants


# --- similarity ---------------------------------------------------------------

def suffix_similarity(
    model: MoveModel, a_ids: Sequence[int], b_ids: Sequence[int]
) -> Dict[str, float]:
    """How close are two suffixes, at token and character level."""
    a_ids, b_ids = list(a_ids), list(b_ids)
    a_text, b_text = model.decode_suffix(a_ids), model.decode_suffix(b_ids)
    positional = (
        sum(1 for x, y in zip(a_ids, b_ids) if x == y) / max(1, len(a_ids))
        if len(a_ids) == len(b_ids)
        else float("nan")
    )
    set_a, set_b = set(a_ids), set(b_ids)
    jaccard = len(set_a & set_b) / max(1, len(set_a | set_b))
    matcher = difflib.SequenceMatcher(None, a_text, b_text)
    match = matcher.find_longest_match(0, len(a_text), 0, len(b_text))
    return {
        "token_positional": positional,
        "token_jaccard": jaccard,
        "char_ratio": matcher.ratio(),
        "longest_common_substring": float(match.size),
    }


def compare_texts(a_text: str, b_text: str) -> Dict[str, float]:
    """Text-level comparison, valid ACROSS tokenisers (different models)."""
    matcher = difflib.SequenceMatcher(None, a_text, b_text)
    match = matcher.find_longest_match(0, len(a_text), 0, len(b_text))
    words_a, words_b = set(a_text.split()), set(b_text.split())
    return {
        "char_ratio": matcher.ratio(),
        "longest_common_substring": float(match.size),
        "word_jaccard": len(words_a & words_b) / max(1, len(words_a | words_b)),
    }


def compare_runs(run_dirs: Sequence[str]) -> str:
    """Pairwise suffix comparison across runs (different models or attackers)."""
    entries = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, "gcg_best_suffix.json")
        if not os.path.exists(path):
            log.warning("no suffix in %s", run_dir)
            continue
        payload = io_utils.read_json(path)
        entries.append(
            {
                "run": os.path.basename(run_dir.rstrip("/")),
                "model": payload.get("model", "?"),
                "suffix": payload["suffix"],
                "n_tokens": len(payload.get("suffix_ids", [])),
            }
        )
    if len(entries) < 2:
        return "need at least two runs with a saved suffix\n"
    lines = ["Suffixes", ""]
    for e in entries:
        lines.append(f"  {e['run']}  [{e['model']}, {e['n_tokens']} tokens]")
        lines.append(f"    {e['suffix']!r}")
    lines += ["", "Pairwise similarity (character level, tokeniser-independent)", ""]
    header = f"{'pair':<50}{'char_ratio':>12}{'LCS':>7}{'word_jacc':>11}"
    lines += [header, "-" * len(header)]
    for i, a in enumerate(entries):
        for b in entries[i + 1 :]:
            sim = compare_texts(a["suffix"], b["suffix"])
            pair = f"{a['run'][:23]} vs {b['run'][:23]}"
            lines.append(
                f"{pair:<50}{sim['char_ratio']:>12.3f}"
                f"{int(sim['longest_common_substring']):>7d}{sim['word_jaccard']:>11.3f}"
            )
    lines += [
        "",
        "char_ratio ~ 1.0 means the same string; two independent GCG runs on",
        "unrelated models landing on a similar string would be a strong claim,",
        "so treat a low ratio as the expected outcome, not as a failure.",
    ]
    return "\n".join(lines)


# --- the study ----------------------------------------------------------------

def run_perturbation_study(
    cfg,
    run_id: Optional[str] = None,
    suffix_override: Optional[str] = None,
    model: Optional[MoveModel] = None,
    draws: int = 3,
) -> Dict[str, object]:
    """Score every perturbation of the suffix on the held-out pools."""
    if run_id:
        cfg.run.run_id = run_id
    run_dir = cfg.run_dir()
    io_utils.setup_logging(run_dir)

    suffix_text = suffix_override
    suffix_ids: Optional[List[int]] = None
    best_json = os.path.join(run_dir, "gcg_best_suffix.json")
    if suffix_text is None and os.path.exists(best_json):
        payload = io_utils.read_json(best_json)
        suffix_text, suffix_ids = payload["suffix"], list(payload["suffix_ids"])
    if suffix_text is None:
        raise FileNotFoundError(
            f"no suffix in {run_dir} and none given with --suffix"
        )

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
    if suffix_ids is None:
        suffix_ids = model.encode_suffix(suffix_text)
    log.info("perturbing %r (%d tokens)", suffix_text, len(suffix_ids))

    datasets = build_datasets(cfg)
    pools = build_pools(model, datasets, cfg, splits=("test",))["test"]
    q_base = model.q_batch(pools["independent"].base_prompts, ())["q"].float()

    rng = np.random.default_rng(cfg.run.seed)
    variants = make_variants(
        model, suffix_ids, rng, draws=draws, allow_non_ascii=cfg.gcg.allow_non_ascii
    )
    log.info("scoring %d variants", len(variants))

    rows: List[Dict[str, object]] = []
    fields = (
        "name", "kind", "draw", "magnitude", "n_tokens",
        "token_positional", "token_jaccard", "char_ratio",
        "model_mi", "help_rate", "util_kl", "validity", "entropy", "suffix",
    )
    csv_log = io_utils.CSVLog(os.path.join(run_dir, "robustness_perturbation.csv"), fields)
    for variant in variants:
        colluding = scalar_metrics(
            measure_pool(model, pools["colluding"], variant.suffix_ids)
        )
        independent = scalar_metrics(
            measure_pool(model, pools["independent"], variant.suffix_ids, q_base=q_base)
        )
        sim = suffix_similarity(model, suffix_ids, variant.suffix_ids)
        row = {
            "name": variant.name,
            "kind": variant.kind,
            "draw": variant.draw,
            "magnitude": round(variant.magnitude, 4),
            "n_tokens": variant.n_tokens,
            "token_positional": round(sim["token_positional"], 4),
            "token_jaccard": round(sim["token_jaccard"], 4),
            "char_ratio": round(sim["char_ratio"], 4),
            "model_mi": round(colluding["model_mi"], 6),
            "help_rate": round(colluding["help_rate"], 4),
            "util_kl": round(independent["util_kl"], 6),
            "validity": round(independent["validity"], 4),
            "entropy": round(independent["entropy"], 4),
            "suffix": model.decode_suffix(variant.suffix_ids),
        }
        rows.append(row)
        csv_log.write(row)
        log.info(
            "%-18s MI=%.5f help=%.3f util_kl=%.5f char_sim=%.3f",
            variant.name,
            row["model_mi"],
            row["help_rate"],
            row["util_kl"],
            row["char_ratio"],
        )
    csv_log.close()

    summary = summarise_perturbations(rows)
    io_utils.write_text(os.path.join(run_dir, "robustness_perturbation.txt"), summary + "\n")
    log.info("perturbation study\n%s", summary)
    from . import plots

    figures = plots.robustness_scatter(run_dir)
    return {"run_dir": run_dir, "rows": rows, "summary": summary, "figures": figures}


def summarise_perturbations(rows: Sequence[Dict[str, object]]) -> str:
    """Mean MI per perturbation kind, ordered from gentlest to harshest."""
    by_kind: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_kind.setdefault(str(row["kind"]), []).append(row)
    original = next((r for r in rows if r["kind"] == "original"), None)
    none_row = next((r for r in rows if r["kind"] == "none"), None)
    header = f"{'perturbation':<16}{'n':>4}{'MI':>10}{'help':>8}{'utilKL':>10}{'char_sim':>10}"
    lines = [header, "-" * len(header)]
    order = sorted(by_kind, key=lambda k: -float(np.mean([r["char_ratio"] for r in by_kind[k]])))
    for kind in order:
        group = by_kind[kind]
        lines.append(
            f"{kind:<16}{len(group):>4}"
            f"{np.mean([r['model_mi'] for r in group]):>10.5f}"
            f"{np.mean([r['help_rate'] for r in group]):>8.3f}"
            f"{np.mean([r['util_kl'] for r in group]):>10.5f}"
            f"{np.mean([r['char_ratio'] for r in group]):>10.3f}"
        )
    if original is not None and none_row is not None:
        gained = float(none_row["model_mi"]) - float(original["model_mi"])
        lines += [
            "",
            f"no defence MI = {none_row['model_mi']:.5f}, suffix MI = {original['model_mi']:.5f} "
            f"(the defence is worth {gained:.5f} nats)",
            "A perturbation that keeps MI near the suffix row preserves the defence;",
            "one that returns to the no-defence row destroys it.",
        ]
    return "\n".join(lines)
