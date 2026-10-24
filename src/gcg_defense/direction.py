"""The "collusion direction": diff-in-means in the residual stream.

Following Arditi et al. (2024), "Refusal in Language Models Is Mediated by a
Single Direction": take the difference of the mean last-token residual stream
between colluding and independent prompts at each layer, normalise it, and pick
the layer where the two conditions separate best (Cohen's d of the projection).

The saved file carries everything the other modules need:
    direction        [d] unit vector r_hat
    layer            the residual-stream layer it was measured at
    mu_independent   mean <x, r_hat> on independent prompts, no suffix
    std_independent  its standard deviation -- the unit in which the steering
                     baseline expresses alpha (alpha = 2 means 2 std)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

from . import io_utils
from .model_utils import MoveModel
from .pools import build_pool
from .prompts import build_datasets

log = logging.getLogger(__name__)


def compute_collusion_direction(
    cfg,
    model: Optional[MoveModel] = None,
    out_path: Optional[str] = None,
) -> Dict[str, object]:
    """Diff-in-means direction over all layers; saves the best layer's vector."""
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
    datasets = build_datasets(cfg)
    colluding = build_pool(
        model, "colluding", datasets.train["colluding"], cfg.prompts.suffix_position
    )
    independent = build_pool(
        model, "independent", datasets.train["independent"], cfg.prompts.suffix_position
    )
    best = None
    rows = []
    # One forward pass over each pool, all layers at once.
    all_c = model.all_layer_acts(colluding.base_prompts, ())
    all_i = model.all_layer_acts(independent.base_prompts, ())
    for layer in range(all_c.shape[1]):
        acts_c = all_c[:, layer, :]
        acts_i = all_i[:, layer, :]
        diff = acts_c.mean(0) - acts_i.mean(0)
        norm = diff.norm()
        if float(norm) == 0.0:
            continue
        direction = (diff / norm).float()
        proj_c = acts_c.float() @ direction
        proj_i = acts_i.float() @ direction
        pooled = torch.sqrt(
            (proj_c.var(unbiased=True) + proj_i.var(unbiased=True)) / 2
        )
        cohens_d = float((proj_c.mean() - proj_i.mean()) / pooled.clamp_min(1e-9))
        rows.append({"layer": layer, "cohens_d": cohens_d, "norm": float(norm)})
        log.info("layer %2d | ||diff||=%.3f Cohen's d=%.2f", layer, float(norm), cohens_d)
        if best is None or cohens_d > best["cohens_d"]:
            best = {
                "layer": layer,
                "cohens_d": cohens_d,
                "direction": direction.cpu(),
                "mu_independent": float(proj_i.mean()),
                "std_independent": float(proj_i.std(unbiased=True)),
                "mu_colluding": float(proj_c.mean()),
            }
    if best is None:
        raise RuntimeError("no usable collusion direction found")
    out_path = out_path or os.path.join(
        cfg.run.out_dir, "collusion_direction.pt"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    gap_in_std = (best["mu_colluding"] - best["mu_independent"]) / max(
        best["std_independent"], 1e-9
    )
    payload = {
        "direction": best["direction"],
        "gap_in_std": float(gap_in_std),
        "layer": best["layer"],
        "mu_independent": best["mu_independent"],
        "std_independent": best["std_independent"],
        "mu_colluding": best["mu_colluding"],
        "cohens_d": best["cohens_d"],
        "model": cfg.model.name,
        "n_contexts": colluding.n_contexts,
        "per_layer": rows,
    }
    torch.save(payload, out_path)
    io_utils.write_json(
        out_path.replace(".pt", ".json"),
        {k: v for k, v in payload.items() if k != "direction"},
    )
    log.info(
        "collusion direction: layer %d (Cohen's d=%.2f), mu_indep=%.3f std=%.4f, "
        "colluding prompts sit %.1f std away -> %s",
        best["layer"],
        best["cohens_d"],
        best["mu_independent"],
        best["std_independent"],
        gap_in_std,
        out_path,
    )
    return {
        "path": out_path,
        "layer": best["layer"],
        "cohens_d": best["cohens_d"],
        "mu_independent": best["mu_independent"],
        "std_independent": best["std_independent"],
        "gap_in_std": float(gap_in_std),
    }


def load_direction_file(path: str, device=None):
    """Load r_hat (re-normalised), its layer and the independent-prompt stats."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    direction = payload["direction"].float()
    direction = direction / direction.norm()
    if device is not None:
        direction = direction.to(device)
    return (
        direction,
        int(payload["layer"]),
        float(payload.get("mu_independent", 0.0)),
        float(payload.get("std_independent", 1.0)),
        float(payload.get("gap_in_std", float("nan"))),
    )
