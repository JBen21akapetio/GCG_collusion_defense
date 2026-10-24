"""Model-level checks: move tokens, suffix re-tokenisation, path equivalence.

Marked `requires_model`: these download a small instruct model and run it on
CPU in float32.
"""

import pytest
import torch

from gcg_defense import prompts
from gcg_defense.model_utils import resolve_move_tokens
from gcg_defense.pools import build_pool

pytestmark = pytest.mark.requires_model


def test_move_tokens_are_distinct_and_reported(small_model):
    tokens = small_model.move_tokens
    assert len(tokens.ids) == 3
    assert len(set(tokens.ids)) == 3
    assert tokens.variant in ("plain", "leading_space", "first_token")
    # The answer follows the assistant header directly, so the bare word (or
    # its first token) is the right variant, never the space-prefixed one.
    assert tokens.variant != "leading_space"
    assert all(piece.strip() for piece in tokens.pieces)
    print(tokens.report())


def test_move_tokens_are_the_first_tokens_of_the_move_words(small_model):
    tok = small_model.tokenizer
    for move, tid in zip(small_model.moves, small_model.move_tokens.ids):
        assert tok.encode(move, add_special_tokens=False)[0] == tid


def test_resolve_move_tokens_is_stable(small_model):
    again = resolve_move_tokens(small_model.tokenizer, small_model.moves)
    assert again.ids == small_model.move_tokens.ids


def test_suffix_survives_retokenisation(small_model):
    good = "! " * 20
    ids = small_model.encode_suffix(good.strip())
    assert small_model.suffix_roundtrips(ids)
    # A random mid-word byte sequence generally does not round-trip; the GCG
    # filter is what keeps such candidates out of the search.
    assert small_model.suffix_roundtrips(small_model.encode_suffix("hello world"))


def test_allowed_token_mask_excludes_specials_and_non_ascii(small_model):
    mask = small_model.allowed_token_mask(allow_non_ascii=False)
    assert mask.dtype == torch.bool
    assert mask.numel() == small_model.vocab_size
    assert 0.1 < mask.float().mean().item() < 1.0
    for tid in small_model.tokenizer.all_special_ids:
        assert not bool(mask[tid])
    for tid in small_model.move_tokens.ids:
        assert bool(mask[tid])


@pytest.fixture(scope="module")
def pool(small_model):
    ctxs = prompts.build_contexts("colluding", 2, 1, __import__("numpy").random.default_rng(0))
    return build_pool(small_model, "colluding", ctxs, "system_end")


def test_q_is_a_distribution_and_validity_is_a_probability(small_model, pool):
    out = small_model.q_batch(pool.prompts, ())
    q, validity = out["q"], out["validity"]
    assert q.shape == (len(pool.prompts), 3)
    assert torch.allclose(q.sum(-1), torch.ones(len(pool.prompts)), atol=1e-5)
    assert (validity >= 0).all() and (validity <= 1).all()
    assert q.dtype == torch.float32 and validity.dtype == torch.float32


def test_candidate_path_matches_batch_path(small_model, pool):
    """q_candidates (prefix KV cache) must agree with the plain forward."""
    suffixes = torch.tensor(
        [small_model.encode_suffix("! " * 5)[:5], small_model.encode_suffix("x y z a b")[:5]]
    )
    prompt = pool.prompts[0]
    cand = small_model.q_candidates(prompt, suffixes)
    for i, suffix in enumerate(suffixes):
        ref = small_model.q_batch([prompt], suffix.tolist())
        assert torch.allclose(cand["q"][i], ref["q"][0], atol=2e-3)
        assert torch.allclose(cand["validity"][i], ref["validity"][0], atol=2e-3)


def test_onehot_path_matches_batch_path(small_model, pool):
    """The differentiable one-hot path must agree with the token-id path."""
    suffix = small_model.encode_suffix("! " * 5)[:5]
    one_hot = torch.zeros(len(suffix), small_model.vocab_size)
    for i, tid in enumerate(suffix):
        one_hot[i, tid] = 1.0
    out_onehot = small_model.q_batch_onehot(pool.prompts, one_hot)
    out_ids = small_model.q_batch(pool.prompts, suffix)
    assert torch.allclose(out_onehot["q"], out_ids["q"], atol=2e-3)


def test_gradient_flows_to_the_one_hot(small_model, pool):
    from gcg_defense import loss

    suffix = small_model.encode_suffix("! " * 5)[:5]
    one_hot = torch.zeros(len(suffix), small_model.vocab_size, requires_grad=True)
    with torch.no_grad():
        for i, tid in enumerate(suffix):
            one_hot[i, tid] = 1.0
    out = small_model.q_batch_onehot(pool.prompts, one_hot)
    q = pool.reshape(out["q"])
    loss.mutual_information(q).mean().backward()
    assert one_hot.grad is not None
    assert torch.isfinite(one_hot.grad).all()
    assert one_hot.grad.abs().sum() > 0


def test_residual_stream_can_be_read(small_model, pool):
    layer = small_model.n_layers // 2
    out = small_model.q_batch(pool.prompts, (), want_layer=layer)
    assert out["acts"].shape == (len(pool.prompts), small_model.hidden_size)
    assert torch.isfinite(out["acts"]).all()


def test_candidate_filter_keeps_most_candidates(small_model):
    """Regression: space cleanup in decode() used to reject 100% of candidates."""
    import numpy as np

    from gcg_defense.gcg import filter_candidates, sample_candidates

    suffix = torch.tensor(small_model.encode_suffix(("! " * 20).rstrip()))
    allowed = small_model.allowed_token_mask(allow_non_ascii=False)
    grad = torch.randn(suffix.numel(), small_model.vocab_size)
    cands = sample_candidates(
        suffix, grad, 256, 128, 1, allowed, np.random.default_rng(0)
    )
    kept = filter_candidates(small_model, cands, keep_first=suffix)
    assert kept.shape[0] / cands.shape[0] > 0.5
    for row in kept:
        assert small_model.suffix_roundtrips([int(t) for t in row])


def test_init_suffix_has_exactly_k_tokens(small_model, cfg):
    from gcg_defense.gcg import init_suffix_ids

    ids = init_suffix_ids(small_model, cfg)
    assert len(ids) == cfg.gcg.suffix_len
    assert small_model.suffix_roundtrips(ids)


def test_activation_hooks_change_q_and_are_removed_afterwards(small_model, pool):
    """A leaked forward hook would silently corrupt every later measurement."""
    from gcg_defense.baselines import activation_steering, directional_ablation

    direction = torch.randn(small_model.hidden_size)
    direction = direction / direction.norm()
    before = small_model.q_batch(pool.prompts, ())["q"]

    with directional_ablation(small_model, direction):
        ablated = small_model.q_batch(pool.prompts, ())["q"]
    after = small_model.q_batch(pool.prompts, ())["q"]
    assert not torch.allclose(before, ablated, atol=1e-4)
    assert torch.allclose(before, after, atol=1e-5)

    layer = small_model.n_layers // 2
    with activation_steering(small_model, direction, alpha=8.0, std=1.0, layer=layer):
        steered = small_model.q_batch(pool.prompts, ())["q"]
    assert not torch.allclose(before, steered, atol=1e-4)
    assert torch.allclose(before, small_model.q_batch(pool.prompts, ())["q"], atol=1e-5)

    with activation_steering(small_model, direction, alpha=0.0, std=1.0, layer=layer):
        unsteered = small_model.q_batch(pool.prompts, ())["q"]
    assert torch.allclose(before, unsteered, atol=1e-5)


def test_random_suffix_baseline_round_trips(small_model):
    import numpy as np

    from gcg_defense.baselines import random_suffix_ids

    ids = random_suffix_ids(small_model, 20, np.random.default_rng(0))
    assert len(ids) == 20
    assert small_model.suffix_roundtrips(ids)


def test_empty_suffix_costs_no_utility(small_model, pool):
    """L_util must satisfy `prompt + "" == prompt`: an empty suffix (no defence,
    or a hook-only baseline) must not be charged for the newline that would
    introduce a suffix."""
    import numpy as np

    from gcg_defense import prompts as prompt_mod
    from gcg_defense.measure import measure_pool
    from gcg_defense.pools import build_pool

    contexts = prompt_mod.build_contexts(
        "independent", 2, 1, np.random.default_rng(0)
    )
    indep = build_pool(small_model, "independent", contexts, "system_end")
    q_base = small_model.q_batch(indep.base_prompts, ())["q"].float()

    empty = measure_pool(small_model, indep, (), q_base=q_base)
    assert empty["util_kl"] == pytest.approx(0.0, abs=1e-6)

    # A real suffix does pay, and the slotted prompt differs from the clean one.
    suffix = small_model.encode_suffix("! ! ! ! !")
    with_suffix = measure_pool(small_model, indep, suffix, q_base=q_base)
    assert with_suffix["util_kl"] > 1e-4
