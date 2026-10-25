"""Perturbations du suffixe, similarité, et split d'instructions paramétrable."""

import numpy as np
import pytest
import torch

from gcg_defense import robustness
from gcg_defense.plots import perturbation_family
from gcg_defense.prompts import COLLUSION_INSTRUCTIONS, instruction_pool

VOCAB = 200


class FakeModel:
    """Assez de modèle pour exercer la logique de perturbation."""

    def __init__(self):
        self.vocab_size = VOCAB

    def allowed_token_mask(self, allow_non_ascii=False):
        mask = torch.zeros(VOCAB, dtype=torch.bool)
        mask[10:150] = True
        return mask

    def decode_suffix(self, ids):
        return " ".join(f"t{i}" for i in ids)

    def encode_suffix(self, text):
        out = []
        for piece in text.split():
            digits = "".join(c for c in piece if c.isdigit())
            out.append(int(digits) if digits else 11)
        return out


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def suffix():
    return list(range(20, 40))


def test_variants_cover_every_family(model, suffix):
    variants = robustness.make_variants(
        model, suffix, np.random.default_rng(0), draws=2
    )
    kinds = {v.kind for v in variants}
    assert "original" in kinds
    assert "shuffle" in kinds
    assert "reverse" in kinds
    assert "truncate" in kinds
    assert "random" in kinds
    assert "none" in kinds
    assert any(k.startswith("replace_") for k in kinds)
    assert any(k.startswith("delete_") for k in kinds)


def test_original_variant_is_untouched(model, suffix):
    variants = robustness.make_variants(model, suffix, np.random.default_rng(0))
    original = next(v for v in variants if v.kind == "original")
    assert original.suffix_ids == suffix


def test_replace_changes_exactly_k_positions(model, suffix):
    variants = robustness.make_variants(
        model, suffix, np.random.default_rng(0), draws=4, replace_counts=(3,)
    )
    for variant in (v for v in variants if v.kind == "replace_3"):
        differing = sum(1 for a, b in zip(suffix, variant.suffix_ids) if a != b)
        assert differing <= 3           # un remplacement peut retomber sur le même id
        assert len(variant.suffix_ids) == len(suffix)


def test_delete_shortens_by_k(model, suffix):
    variants = robustness.make_variants(
        model, suffix, np.random.default_rng(0), draws=2, delete_counts=(5,)
    )
    for variant in (v for v in variants if v.kind == "delete_5"):
        assert len(variant.suffix_ids) == len(suffix) - 5


def test_shuffle_keeps_the_same_multiset(model, suffix):
    """La variante décisive : mêmes tokens, ordre différent."""
    variants = robustness.make_variants(model, suffix, np.random.default_rng(0), draws=3)
    shuffles = [v for v in variants if v.kind == "shuffle"]
    assert shuffles
    for variant in shuffles:
        assert sorted(variant.suffix_ids) == sorted(suffix)
    assert any(v.suffix_ids != suffix for v in shuffles)


def test_variants_only_use_allowed_tokens(model, suffix):
    allowed = set(model.allowed_token_mask().nonzero().flatten().tolist())
    variants = robustness.make_variants(model, suffix, np.random.default_rng(0))
    for variant in variants:
        if variant.kind in ("random",):
            assert set(variant.suffix_ids) <= allowed


def test_similarity_of_a_suffix_with_itself_is_one(model, suffix):
    sim = robustness.suffix_similarity(model, suffix, suffix)
    assert sim["token_positional"] == pytest.approx(1.0)
    assert sim["token_jaccard"] == pytest.approx(1.0)
    assert sim["char_ratio"] == pytest.approx(1.0)


def test_similarity_decreases_with_distance(model, suffix):
    one = list(suffix)
    one[0] = 199
    many = [199] * len(suffix)
    close = robustness.suffix_similarity(model, suffix, one)
    far = robustness.suffix_similarity(model, suffix, many)
    assert close["token_positional"] > far["token_positional"]
    assert close["char_ratio"] > far["char_ratio"]


def test_similarity_handles_different_lengths(model, suffix):
    sim = robustness.suffix_similarity(model, suffix, suffix[:5])
    assert np.isnan(sim["token_positional"])     # non défini si longueurs différentes
    assert 0.0 < sim["char_ratio"] < 1.0


def test_compare_texts_is_tokeniser_independent():
    same = robustness.compare_texts("abc def", "abc def")
    assert same["char_ratio"] == pytest.approx(1.0)
    assert same["word_jaccard"] == pytest.approx(1.0)
    different = robustness.compare_texts("abc def", "xyz uvw")
    assert different["char_ratio"] < 0.4
    assert different["word_jaccard"] == pytest.approx(0.0)


def test_summary_orders_and_reports_the_reference_rows():
    rows = [
        {"kind": "original", "model_mi": 0.05, "help_rate": 0.36, "util_kl": 0.01, "char_ratio": 1.0},
        {"kind": "replace_1", "model_mi": 0.07, "help_rate": 0.40, "util_kl": 0.02, "char_ratio": 0.9},
        {"kind": "none", "model_mi": 0.63, "help_rate": 0.82, "util_kl": 0.0, "char_ratio": 0.0},
    ]
    text = robustness.summarise_perturbations(rows)
    assert "original" in text and "replace_1" in text
    assert "no defence MI = 0.63000" in text
    assert "0.58000" in text  # 0.63 - 0.05, ce que vaut la défense


def test_plot_family_mapping():
    assert perturbation_family("replace_5") == "token"
    assert perturbation_family("shuffle") == "token"
    assert perturbation_family("truncate") == "token"
    assert perturbation_family("char_edit") == "char"
    assert perturbation_family("random") == "control"
    assert perturbation_family("none") == "control"


def test_explicit_train_instruction_ids_flip_the_split():
    """Expérience 3 : entraîner sur l'autre moitié des instructions."""
    total = len(COLLUSION_INSTRUCTIONS)
    train = instruction_pool("colluding", "train", 8, train_ids=[8, 9, 10, 11, 12])
    test = instruction_pool("colluding", "test", 8, train_ids=[8, 9, 10, 11, 12])
    assert train == [8, 9, 10, 11, 12]
    assert test == list(range(8))
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == total


def test_default_split_is_unchanged_without_explicit_ids():
    assert instruction_pool("colluding", "train", 8) == list(range(8))
    assert instruction_pool("colluding", "test", 8) == list(
        range(8, len(COLLUSION_INSTRUCTIONS))
    )
